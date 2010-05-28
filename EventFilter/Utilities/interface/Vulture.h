#ifndef EVENTFILTER_UTILITIES_VULTURE_H
#define EVENTFILTER_UTILITIES_VULTURE_H



#include <string>
#include <vector>

#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>

#include <toolbox/lang/Class.h>

#include "EventFilter/Utilities/interface/MasterQueue.h"
#include "EventFilter/Utilities/interface/SlaveQueue.h"

namespace toolbox{
  namespace task{
    class WorkLoop;
    class ActionSignature;
  }
}

namespace evf{

  class CurlPoster;
  
  static const int VULTURE_START_MESSAGE_URL_SIZE = 128;

  struct vulture_start_message{
    char url_[VULTURE_START_MESSAGE_URL_SIZE];
    int  run_;
  };

  class Vulture : public toolbox::lang::Class {

  public:

    Vulture(bool);
    virtual ~Vulture();
    pid_t makeProcess();
    int start(std::string,int=0);
    int stop();
    int  hasStarted();
    int  hasStopped();
    pid_t kill();
    void retrieve_corefile(char *, char *, uint64_t);


  private:
    
    static const int vulture_queue_offset = 400;
    void startProwling();
    bool control(toolbox::task::WorkLoop*);
    bool prowling(toolbox::task::WorkLoop*);
    void analyze();

    static const std::string FS;
    toolbox::task::WorkLoop         *wlCtrl_;      
    toolbox::task::ActionSignature  *asCtrl_;
    bool                             running_;
    toolbox::task::WorkLoop         *wlProwl_;      
    toolbox::task::ActionSignature  *asProwl_;
    bool                             prowling_;
    std::string                      iDieUrl_;
    bool                             updateMode_;
    pid_t                            vulturePid_;
    DIR                             *tmp_;
    std::vector<std::string>         currentCoreList_;
    time_t                           lastUpdate_;
    unsigned int                     newCores_;
    CurlPoster                      *poster_;
    MasterQueue                     *mq_;
    SlaveQueue                      *sq_;
    int                              started_;
    int                              stopped_;
    bool                             handicapped_;
  };
}
#endif
