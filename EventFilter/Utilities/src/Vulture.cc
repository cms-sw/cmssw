
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

  Vulture::Vulture(std::string &url, bool push) :
    wlPrey_(0)
    , asPrey_(0)
    , preying_(false)
    , iDieUrl_(url)
    , updateMode_(push)
    , vulturePid_(0)
    , tmp_(0)
    , newCores_(0)
    , poster_(0)
  {
    FILE *outf = fopen("/tmp/vulture.cmd","w");
    fprintf(outf,"where\n");
    fclose(outf);
  }

  pid_t Vulture::start(int run){
    pid_t retval = fork();
    if(retval==0){ // we are in the forked process
      try{
	pt::PeerTransport * ptr =
	  pt::getPeerTransportAgent()->getPeerTransport("http","soap",pt::Receiver);
	delete ptr;
      }
      catch (pt::exception::PeerTransportNotFound & e ){
	//do nothing here since we don't know what to do... ?
      }
      std::cout << "vulture is here 1" << std::endl;
      int success = prctl( PR_SET_DUMPABLE, 0 );
      if(success != 0) ::exit(-1);
      success = prctl( PR_SET_PDEATHSIG, SIGKILL );
      if(success != 0) ::exit(-1);
      std::cout << "vulture is here 2" << std::endl;

      if(success != 0) ::exit(-1);
      std::cout << "vulture is here 3" << std::endl;
      tmp_ = opendir("/tmp");
      poster_ = new CurlPoster(iDieUrl_);
      if(poster_->check(run)){
	try{
	  std::cout << "vulture is here 4" << std::endl;
	  startPreying();
	}
	catch(evf::Exception &e)
	  {
	    ::exit(-1);
	  }
      }
      else{
	::exit(0);
      }
      std::cout << "vulture is running" << std::endl;
      success = prctl ( PR_SET_NAME , "vulture");
    }
    else
      {
	vulturePid_ = retval;
      }
    return retval;
  }
  
  pid_t Vulture::stop()
  {
    preying_ = false;
    std::cout << "Sending SIGKILL to vulture on pid " 
	      << vulturePid_  << std::endl;
    int retval = kill (vulturePid_, SIGKILL);
    std::cout << "trying to kill Vulture returned " << retval << std::endl;
    int sl;
    pid_t killedOrNot = waitpid(vulturePid_,&sl,WNOHANG);
    std::cout << "waitpid on Vulture returned "  << killedOrNot << std::endl;
    vulturePid_ = 0;
    char messageDie[5];
    sprintf(messageDie,"Dead");
    poster_ = new CurlPoster(iDieUrl_);
    try{
      poster_->postString(messageDie,5,0);
    }
    catch(evf::Exception &e){
      //do nothing just swallow the exception
    }
    delete poster_;
    poster_=0;
    std::cout <<"Sending Dead Signal to iDie" << std::endl;
    return killedOrNot;
  }
  
  void Vulture::startPreying()
  {
    timeval now;
    gettimeofday(&now,0);
    lastUpdate_ = now.tv_sec;
    preying_ = true;
    try {
      wlPrey_=
	toolbox::task::getWorkLoopFactory()->getWorkLoop("Prey",
							 "waiting");
      if (!wlPrey_->isActive()) wlPrey_->activate();
      
      asPrey_ = toolbox::task::bind(this,&Vulture::preying,
					 "Prey");
      wlPrey_->submit(asPrey_);
    }
    catch (xcept::Exception& e) {
      std::string msg = "Failed to start workloop 'Prey'.";
      XCEPT_RETHROW(evf::Exception,msg,e);
    }

  }
  bool Vulture::preying(toolbox::task::WorkLoop*wl)
  {
    std::cout << "Vulture::swoop" << std::endl;

    if(!preying_) return false;
    newCores_ = 0;
    struct stat filestat;    
//     stat(FS.c_str(),&filestat);
//     if(filestat.st_mtime < lastUpdate_){ 
//       std::cout << "tmp has not changed !!!" << std::endl;
//       ::sleep(60); 
//       return true;
//     }
    timeval now;

    gettimeofday(&now,0);
    // examine /tmp looking for new coredumps
    dirent *dirp;
    while((dirp = readdir(tmp_))!=0){
      std::cout << "found file " << dirp->d_name << std::endl;
      if(strncmp(dirp->d_name,"core",4)==0){
	stat(dirp->d_name,&filestat);
	std::cout << "Found corefile " <<  dirp->d_name << " lastmod " 
		  << filestat.st_mtime << " lastup " << lastUpdate_
		  << std::endl;
	if(filestat.st_mtime > lastUpdate_){
	  std::cout << "Found new corefile " << dirp->d_name << std::endl;
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
    std::cout << "Vulture::swoop completed" << std::endl;
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
      std::string filePathAndName = "/tmp";
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
      int rch = chmod(filePathAndName.c_str(),0777);
      if(rch != 0)
	std::cout << "ERROR: couldn't change corefile access privileges" << std::endl;
      unsigned int ipid = (unsigned int)atoi(pid.c_str());
      poster_->postString(cmdout.c_str(),cmdout.length(),ipid); 
      
    }
  }
}

