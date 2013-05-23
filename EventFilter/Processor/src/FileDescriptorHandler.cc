#include "FileDescriptorHandler.h"

#include <iostream>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <stdlib.h>


namespace evf{

FileDescriptorHandler::FileDescriptorHandler(){

  //find all socket file descriptors inherited from parent process and close them

  pid_t pid = ::getpid();
  std::ostringstream ost;
  ost << "/proc/" << pid << "/fd/";
  DIR *dir = opendir(ost.str().c_str());
  dirent *de; 
  struct stat buf;
  while((de = readdir(dir))!=0){
    char *name = de->d_name;    
    std::string path = ost.str()+name;
    stat(path.c_str(),&buf);
    if(S_ISSOCK(buf.st_mode)){
      int fd = atoi(name);
      close(fd);
    }
  }
  closedir(dir);
}

}
