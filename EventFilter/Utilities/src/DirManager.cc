#include "EventFilter/Utilities/interface/DirManager.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

namespace evf {

  unsigned int DirManager::findHighestRun() {
    DIR *dir = opendir(dir_.c_str());
    struct dirent *buf;
    int maxrun = 0;
    while ((buf = readdir(dir))) {
      std::string dirnameNum = buf->d_name;
      if (dirnameNum.find("run") != std::string::npos)
        dirnameNum = dirnameNum.substr(3, std::string::npos);
      if (atoi(dirnameNum.c_str()) > maxrun) {
        maxrun = atoi(dirnameNum.c_str());
      }
    }
    closedir(dir);
    return maxrun;
  }
  std::string DirManager::findHighestRunDir() {
    std::string retval = dir_ + "/";
    std::string tmpdir;
    DIR *dir = opendir(dir_.c_str());
    struct dirent *buf;
    int maxrun = 0;
    while ((buf = readdir(dir))) {
      std::string dirnameNum = buf->d_name;
      if (dirnameNum.find("run") != std::string::npos)
        dirnameNum = dirnameNum.substr(3, std::string::npos);
      if (atoi(dirnameNum.c_str()) > maxrun) {
        tmpdir = buf->d_name;
        maxrun = atoi(dirnameNum.c_str());
      }
    }
    closedir(dir);
    retval += tmpdir;
    return retval;
  }

  std::string DirManager::findRunDir(unsigned int run) {
    std::string retval = dir_ + "/";
    std::string tmpdir = "";
    DIR *dir = opendir(dir_.c_str());
    struct dirent *buf;
    while ((buf = readdir(dir))) {
      std::string dirnameNum = buf->d_name;
      if (dirnameNum.find("run") != std::string::npos)
        dirnameNum = dirnameNum.substr(3, std::string::npos);
      if ((unsigned int)atoi(dirnameNum.c_str()) == run) {
        tmpdir = buf->d_name;
        break;
      }
    }
    closedir(dir);
    if (tmpdir.empty())
      throw cms::Exception("LogicError") << "Run Directory for Run " << run << " Not Found";
    retval += tmpdir;
    return retval;
  }
  bool DirManager::checkDirEmpty(std::string &d) {
    int filecount = 0;
    DIR *dir = opendir(d.c_str());
    struct dirent *buf;
    while ((buf = readdir(dir))) {
      filecount++;
    }
    return (filecount == 0);
  }
}  // namespace evf
