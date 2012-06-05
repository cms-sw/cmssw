#ifndef EVF_MODULEWEB_H
#define EVF_MODULEWEB_H

#include "toolbox/lang/Class.h"
#include "xdata/InfoSpace.h"
#include <string>
#include <pthread.h>
#include <semaphore.h>

namespace xgi{
  class Input;
  class Output;
}


namespace evf
{


namespace moduleweb {
  class ForkParams {
    public:
      ForkParams():slotId(-1),restart(0),isMaster(-1){}
      int slotId;
      bool restart;
      int isMaster;
  };
  class ForkInfoObj {
    public:
      ForkInfoObj()
      {
	control_sem_ = new sem_t;
	sem_init(control_sem_,0,0);
	stopCondition=0;
      }
      ~ForkInfoObj()
      {
	sem_destroy(control_sem_);
	delete control_sem_;
      }
      void lock() {if (mst_lock_) pthread_mutex_lock(mst_lock_);}
      void unlock() {if (mst_lock_) pthread_mutex_unlock(mst_lock_);}
      void (*forkHandler) (void *);
      ForkParams forkParams;
      unsigned int stopCondition;
      sem_t *control_sem_;
      pthread_mutex_t * mst_lock_;
      void * fuAddr;
  };
}

  class ModuleWeb : public toolbox::lang::Class
    {
    public:
      ModuleWeb(const std::string &);
      virtual ~ModuleWeb(){}
      virtual void defaultWebPage(xgi::Input *in, xgi::Output *out); 
      virtual void publish(xdata::InfoSpace *) = 0;
      virtual void publishToXmas(xdata::InfoSpace *){};
    protected:
      std::string moduleName_;
    private:
      virtual void openBackDoor(unsigned int timeout_sec = 0, bool * started = 0){};
      virtual void closeBackDoor(){};
      virtual void publishForkInfo(moduleweb::ForkInfoObj *forkInfoObj);
      friend class ModuleWebRegistry;
    };
}
#endif
