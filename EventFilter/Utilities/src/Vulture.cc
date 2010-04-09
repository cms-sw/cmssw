
#include "EventFilter/Utilities/interface/Vulture.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/CurlPoster.h"

// to handle pt file descriptors left open at fork
#include "pt/PeerTransportReceiver.h"
#include "pt/PeerTransportAgent.h"

#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoop.h"
#include "toolbox/task/WorkLoopFactory.h"

#include <unistd.h>
#include <sys/prctl.h>
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

namespace evf{

  const std::string Vulture::FS="/tmp";

  Vulture::Vulture(bool push) :
      wlCtrl_(0)
    , asCtrl_(0)
    , running_(false)
    , wlProwl_(0)
    , asProwl_(0)
    , prowling_(false)
    , iDieUrl_("")
    , updateMode_(push)
    , vulturePid_(0)
    , tmp_(0)
    , newCores_(0)
    , poster_(0)
    , mq_(new MasterQueue(vulture_queue_offset))
    , sq_(0) // this is only defined in the forked process
    , started_(-1)
    , stopped_(-1)
  {
    // create command file for gdb
    FILE *outf = fopen("/tmp/vulture.cmd","w");
    fprintf(outf,"where\n");
    fclose(outf);
    pid_t retval = fork();
    if(retval==0){ // we are in the forked process
      int success = prctl( PR_SET_DUMPABLE, 0 );
      if(success != 0) ::exit(-1);
      success = prctl( PR_SET_PDEATHSIG, SIGKILL );
      if(success != 0) ::exit(-1);
      tmp_ = opendir(FS.c_str());
      success = prctl ( PR_SET_NAME , "vulture");
      try{
	pt::PeerTransport * ptr =
	  pt::getPeerTransportAgent()->getPeerTransport("http","soap",pt::Receiver);
	delete ptr;
      }
      catch (pt::exception::PeerTransportNotFound & e ){
	//do nothing here since we don't know what to do... ?
      }
      freopen("/dev/null","w",stderr);
      sq_ = new SlaveQueue(vulture_queue_offset);
      // start the ctrl workloop
      try {
	wlCtrl_=
	  toolbox::task::getWorkLoopFactory()->getWorkLoop("Ctrll",
							   "waiting");
	if (!wlCtrl_->isActive()) wlCtrl_->activate();
	
	asCtrl_ = toolbox::task::bind(this,&Vulture::control,
				       "Ctrl");
	wlCtrl_->submit(asCtrl_);
      }
      catch (xcept::Exception& e) {
	std::string msg = "Failed to start workloop 'Ctrl'.";
	::exit(-1);
      }
    }
    else{
      vulturePid_ = retval;
    }

  }
  
  Vulture::~Vulture()
  {
    delete mq_;
    if(sq_ != 0) delete sq_;
    if(poster_ != 0) delete poster_;
  }


  int Vulture::hasStarted(){
    if(started_<0){
      MsgBuf msg2(MAX_MSG_SIZE,MSQS_VULTURE_TYPE_ACK);
      try{
	mq_->rcvNonBlocking(msg2);
	started_ = 0;
      }
      catch(evf::Exception &e){
      }
    } else {started_ = 1;}
    return started_;	
  }

  int Vulture::hasStopped(){
    if(stopped_<0){
      MsgBuf msg2(MAX_MSG_SIZE,MSQS_VULTURE_TYPE_ACK);
      try{
	mq_->rcvNonBlocking(msg2);
	stopped_ = 0;
      }
      catch(evf::Exception &e){
      }
    } else {stopped_ = 1;}
    return stopped_;	
  }

  pid_t Vulture::start(std::string url, int run){

    //communicate start-of-run to Vulture
    vulture_start_message stamsg;
    strncpy(stamsg.url_,url.c_str(),url.length()); 
    stamsg.run_ = run;
    MsgBuf msg1(sizeof(vulture_start_message),MSQM_VULTURE_TYPE_STA);
    memcpy(msg1->mtext,&stamsg,sizeof(vulture_start_message));
    mq_->post(msg1);
    stopped_ = -1;
    return vulturePid_;
  }
  
  pid_t Vulture::stop()
  {

    MsgBuf msg1(NUMERIC_MESSAGE_SIZE,MSQM_VULTURE_TYPE_STP);
    mq_->post(msg1);
    started_ = -1;
    return vulturePid_;
  }

 pid_t Vulture::kill() // eventually *could* be called by master app - it isn't now
  {
    ::kill (vulturePid_, SIGKILL);
    int sl;
    pid_t killedOrNot = waitpid(vulturePid_,&sl,WNOHANG);
    vulturePid_ = 0;
    return killedOrNot;
  }

  void Vulture::startProwling()
  {
    timeval now;
    gettimeofday(&now,0);
    lastUpdate_ = now.tv_sec;
    prowling_ = true;
    try {
      wlProwl_=
	toolbox::task::getWorkLoopFactory()->getWorkLoop("Prowl",
							 "waiting");
      if (!wlProwl_->isActive()) wlProwl_->activate();
      
      asProwl_ = toolbox::task::bind(this,&Vulture::prowling,
					 "Prowl");
      wlProwl_->submit(asProwl_);
    }
    catch (xcept::Exception& e) {
      std::string msg = "Failed to start workloop 'Prowl'.";
      XCEPT_RETHROW(evf::Exception,msg,e);
    }

  }

  bool Vulture::control(toolbox::task::WorkLoop*wl)
  {
    MsgBuf msg;
    unsigned long mtype = MSQM_MESSAGE_TYPE_NOP;
    try{mtype = sq_->rcv(msg);}catch(evf::Exception &e){::exit(-1);}
    switch(mtype){
    case MSQM_VULTURE_TYPE_STA:
      {

	vulture_start_message *sta = (vulture_start_message*)msg->mtext;
	if(poster_ == 0) poster_ = new CurlPoster(sta->url_);
	if(poster_->check(sta->run_)){
	  try{
	    startProwling();
	  }
	  catch(evf::Exception &e)
	    {
	      //@EM ToDo generate some message here
	    }	  
	}else{
	  ::exit(0);
	}
      
	break;
      }
    case MSQM_VULTURE_TYPE_STP:
      {
	prowling_ = false;
	break;
      }
    default:
      {
	// do nothing @EM ToDo generate an appropriate error message
      }
    }
    return true;
    
  }

  bool Vulture::prowling(toolbox::task::WorkLoop*wl)
  {

    if(!prowling_){
      char messageDie[5];
      sprintf(messageDie,"Dead");
      
      try{
	poster_->postString(messageDie,5,0);
      }
      catch(evf::Exception &e){
	  //do nothing just swallow the exception
      }
      delete poster_;
      poster_=0;
      
      return false;
    }
    
    newCores_ = 0;
    
    struct stat filestat;    
    
    timeval now;
    gettimeofday(&now,0);
    
    // examine /tmp looking for new coredumps
    dirent *dirp;
    while((dirp = readdir(tmp_))!=0){
      if(strncmp(dirp->d_name,"core",4)==0){
	stat(dirp->d_name,&filestat);
	if(filestat.st_mtime > lastUpdate_){
	  currentCoreList_.push_back(dirp->d_name);
	  newCores_++;
	}
      }
    }
    rewinddir(tmp_);
    lastUpdate_ = now.tv_sec;
    try{
      analyze();
    }
    catch(evf::Exception &e){
      std::cout << "Vulture cannot send to iDie server, bail out " << std::endl;
      return false;
    }
    ::sleep(60);
    return true;
  }

  void Vulture::analyze()
  {
    // do a first analysis of the coredump
    if(newCores_==0) return;
    for(unsigned int i = currentCoreList_.size()-newCores_; 
	i < currentCoreList_.size();
	i++){
      std::string command = "gdb /opt/xdaq/bin/xdaq.exe -batch -x /tmp/vulture.cmd -c /tmp/";
      std::string cmdout;
      command += currentCoreList_[i];
      std::string filePathAndName = FS + "/";
      filePathAndName += currentCoreList_[i];
      std::string pid = 
	currentCoreList_[i].substr(currentCoreList_[i].find_first_of(".")+1,
				   currentCoreList_[i].length());

      FILE *ps = popen(command.c_str(),"r");
      size_t s = 256;
      char *p=new char[s];
      bool filter = false;
      while(getline(&p,&s,ps) != -1){
	if(strncmp("Core",p,4)==0) filter = true;
	if(filter)cmdout += p;
      }
      delete[] p;
      pclose(ps);
      int errsv = 0;
      int rch = chmod(filePathAndName.c_str(),0777);
      if(rch != 0){
	errsv = errno;
	std::cout << "ERROR: couldn't change corefile access privileges -" 
		  << strerror(errsv)<< std::endl;
      }
      unsigned int ipid = (unsigned int)atoi(pid.c_str());
      poster_->postString(cmdout.c_str(),cmdout.length(),ipid); 
      
    }
  }
}

