
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
#ifdef linux
#include <sys/prctl.h>
#endif
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>

#ifdef __APPLE__
/* getline implementation is copied from glibc. */

#ifndef SIZE_MAX
# define SIZE_MAX ((size_t) -1)
#endif
#ifndef SSIZE_MAX
# define SSIZE_MAX ((ssize_t) (SIZE_MAX / 2))
#endif

ssize_t getline (char **lineptr, size_t *n, FILE *fp)
{
    ssize_t result;
    size_t cur_len = 0;

    if (lineptr == NULL || n == NULL || fp == NULL)
    {
	errno = EINVAL;
        return -1;
   }

    if (*lineptr == NULL || *n == 0)
    {
	*n = 120;
        *lineptr = (char *) malloc (*n);
	if (*lineptr == NULL)
	{
	    result = -1;
	    goto end;
	}
    }

    for (;;)
    {
	int i;

	i = getc (fp);
        if (i == EOF)
	{
	    result = -1;
	    break;
	}

	/* Make enough space for len+1 (for final NUL) bytes.  */
	if (cur_len + 1 >= *n)
	{
	    size_t needed_max =
		SSIZE_MAX < SIZE_MAX ? (size_t) SSIZE_MAX + 1 : SIZE_MAX;
	    size_t needed = 2 * *n + 1;   /* Be generous. */
	    char *new_lineptr;

	    if (needed_max < needed)
		needed = needed_max;
	    if (cur_len + 1 >= needed)
	    {
		result = -1;
		goto end;
	    }

	    new_lineptr = (char *) realloc (*lineptr, needed);
	    if (new_lineptr == NULL)
	    {
		result = -1;
		goto end;
	    }

	    *lineptr = new_lineptr;
	    *n = needed;
	}

	(*lineptr)[cur_len] = i;
	cur_len++;

	if (i == '\n')
	    break;
    }
    (*lineptr)[cur_len] = '\0';
    result = cur_len ? (ssize_t) cur_len : result;

end:
    return result;
}
#endif

namespace evf{

  const std::string Vulture::FS="/tmp";

  Vulture::Vulture(bool push) 
    : wlCtrl_(0)
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
    , handicapped_(false)
  {
    // create command file for gdb, if not already there
	std::ifstream vulture("/tmp/vulture.cmd");
	if (!vulture.good())
	{
		FILE *outf = fopen("/tmp/vulture.cmd","w");
		fprintf(outf,"where\n");
		fclose(outf);
	}

  }
  
  Vulture::~Vulture()
  {
    delete mq_;
    if(sq_ != 0) delete sq_;
    if(poster_ != 0) delete poster_;
  }

  pid_t Vulture::makeProcess(){

    pid_t retval = fork();
    if(retval==0){ // we are in the forked process
      int success = -1;
// #ifdef linux
//       success = prctl( PR_SET_DUMPABLE, 0 );
// #endif
      if(success != 0){
	std::cout << "Vulture::could not set process undumpable" << std::endl;
	handicapped_ = true;
      }
#ifdef linux
      success = prctl( PR_SET_PDEATHSIG, SIGKILL );
#endif
      if(success != 0){
	std::cout << "Vulture::could not set process death signal" << std::endl;
	handicapped_ = true;	
      }
      tmp_ = opendir(FS.c_str());
#ifdef linux
      success = prctl ( PR_SET_NAME , "vulture");
#endif
      if(success != 0){
	std::cout << "Vulture::could not set process name" << std::endl;
	handicapped_ = true;	
      }

      try{
	pt::PeerTransport * ptr =
	  pt::getPeerTransportAgent()->getPeerTransport("http","soap",pt::Receiver);
	delete ptr;
      }
      catch (pt::exception::PeerTransportNotFound & e ){
	//do nothing here since we don't know what to do... ?
      }
      //      freopen("/dev/null","w",stderr);
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
	std::cout << "Vulture:constructor - could not start workloop 'Ctrl' for process " << retval << std::endl;
      }
    }
    else{
      vulturePid_ = retval;
    }
    return retval;


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
    strcpy(stamsg.url_,url.c_str()); 
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
    try{mtype = sq_->rcv(msg);}catch(evf::Exception &e){
      std::cout << "Vulture::exception on msgrcv for control, bailing out of control workloop - good bye" << std::endl;
      return false;
    }
    mtype = msg->mtype;
    switch(mtype){
    case MSQM_VULTURE_TYPE_STA:
      {

	vulture_start_message *sta = (vulture_start_message*)msg->mtext;
	if(poster_ == 0) poster_ = new CurlPoster(sta->url_);
	if(poster_->check(sta->run_)){
	  try{
	    startProwling();
	    MsgBuf msg1(0,MSQS_VULTURE_TYPE_ACK) ;
	    sq_->post(msg1);
	  }
	  catch(evf::Exception &e)
	    {
	      std::cout << "Vulture::start - exception in starting prowling workloop " << e.what() << std::endl;
	      //@EM ToDo generate some message here
	    }	  
	}else{
	  std::cout << "Vulture::start - could not contact iDie - chech Url - will not start prowling loop" << std::endl;
	  prowling_ = false;
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
      if(poster_==0){
	std::cout << "Vulture: asked to stop prowling but no poster " 
		  << std::endl;
	return false;
      }
      try{
	poster_->postString(messageDie,5,0,CurlPoster::stack);
      }
      catch(evf::Exception &e){
	  //do nothing just swallow the exception
      }
      std::cout << "Received STOP message, going to delete poster " << std::endl;
//       delete poster_;
//       poster_=0;
      
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
      poster_->postString(cmdout.c_str(),cmdout.length(),ipid, CurlPoster::stack); 
      
    }
  }
}

