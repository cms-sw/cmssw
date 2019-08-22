#ifndef EVFUTILITIESDIRMANAGER_H
#define EVFUTILITIESDIRMANAGER_H

#include <sys/types.h>
#include <dirent.h>

#include <string>

#include <cstdlib>

namespace evf {

  class DirManager {
  public:
    DirManager(std::string &d) : dir_(d) {}
    virtual ~DirManager() {}
    unsigned int findHighestRun();
    std::string findHighestRunDir();
    std::string findRunDir(unsigned int);
    bool checkDirEmpty(std::string &);

  private:
    std::string dir_;  // this is the base dir with all runs in it
  };

}  // namespace evf
#endif
