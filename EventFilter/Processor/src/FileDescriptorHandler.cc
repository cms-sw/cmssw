#include "FileDescriptorHandler.h"

#include <iostream>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <stdlib.h>
#include <unistd.h>

#include <vector>

namespace evf{

FileDescriptorHandler::FileDescriptorHandler(){

  //find all socket file descriptors inherited from parent process and close them

  pid_t pid = ::getpid();
  std::ostringstream ost;
  ost << "/proc/" << pid << "/fd/";
  DIR *dir = opendir(ost.str().c_str());
  dirent *de; 
  struct stat buf;
  std::vector<int> oldfds;
  std::vector<int> newfds;

  while((de = readdir(dir))!=0){
    char *name = de->d_name;    
    std::string path = ost.str()+name;
    stat(path.c_str(),&buf);
    if(S_ISSOCK(buf.st_mode)){
      int fd = atoi(name);
      oldfds.push_back(fd);
      int newfd = dup(fd);
      if(newfd>0) newfds.push_back(newfd);
      else std::cout <<"couldn't duplicate old fd " << fd << std::endl;
    }
  }
  closedir(dir);
  for(unsigned int i = 0; i < oldfds.size(); i++){
    close(oldfds[i]);
    int newfd = dup2(newfds[i],oldfds[i]);
    if(newfd!=oldfds[i]) std::cout <<"couldn't duplicate new fd to old " 
				   << oldfds[i] << std::endl;
    close(newfds[i]);
  }
}

}
