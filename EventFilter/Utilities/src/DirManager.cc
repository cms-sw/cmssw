#include "EventFilter/Utilities/interface/DirManager.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <string>
#include <string_view>

namespace evf {

  unsigned int DirManager::findHighestRun() {
    DIR *dir = opendir(dir_.c_str());
    struct dirent *buf;
    int maxrun = 0;
    while ((buf = readdir(dir))) {
      if (buf->d_type != DT_DIR and buf->d_type != DT_UNKNOWN)
        continue;
      std::string_view dirnameNum = buf->d_name;
      if (dirnameNum.starts_with("run"))
        dirnameNum.remove_prefix(3);
      if (int run = atoi(dirnameNum.data()); run > maxrun) {
        maxrun = run;
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
      if (buf->d_type != DT_DIR and buf->d_type != DT_UNKNOWN)
        continue;
      std::string_view dirnameNum = buf->d_name;
      if (dirnameNum.starts_with("run"))
        dirnameNum.remove_prefix(3);
      if (int run = atoi(dirnameNum.data()); run > maxrun) {
        tmpdir = buf->d_name;
        maxrun = run;
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
      if (buf->d_type != DT_DIR and buf->d_type != DT_UNKNOWN)
        continue;
      std::string_view dirnameNum = buf->d_name;
      if (dirnameNum.starts_with("run"))
        dirnameNum.remove_prefix(3);
      if ((unsigned int)atoi(dirnameNum.data()) == run) {
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

  bool DirManager::checkDirEmpty(const std::string &d) {
    int filecount = 0;
    DIR *dir = opendir(d.c_str());
    while (readdir(dir)) {
      filecount++;
    }
    closedir(dir);
    return (filecount == 0);
  }
}  // namespace evf
