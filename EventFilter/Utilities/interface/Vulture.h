#ifndef EVENTFILTER_UTILITIES_VULTURE_H
#define EVENTFILTER_UTILITIES_VULTURE_H



#include <string>
#include <vector>

#include <sys/types.h>
#include <dirent.h>

#include <toolbox/lang/Class.h>

namespace toolbox{
  namespace task{
    class WorkLoop;
    class ActionSignature;
  }
}

namespace evf{

  class CurlPoster;

  class Vulture : public toolbox::lang::Class {

  public:

    Vulture(std::string &, bool);
    virtual ~Vulture(){}
    pid_t start(int=0);
    int stop();
    void retrieve_corefile(char *, char *, uint64_t);


  private:
    
    void startPreying();
    bool preying(toolbox::task::WorkLoop*);
    void analyze();

    static const std::string FS;
    toolbox::task::WorkLoop         *wlPrey_;      
    toolbox::task::ActionSignature  *asPrey_;
    bool                             preying_;
    std::string                      iDieUrl_;
    bool                             updateMode_;
    pid_t                            vulturePid_;
    DIR                             *tmp_;
    std::vector<std::string>         currentCoreList_;
    time_t                           lastUpdate_;
    unsigned int                     newCores_;
    CurlPoster                      *poster_;
  };
}
#endif
