#ifndef EVF_MODULEWEB_H
#define EVF_MODULEWEB_H

//#include "toolbox/lang/Class.h"
//#include "xdata/InfoSpace.h"
#include <string>
#include <pthread.h>
#include <semaphore.h>
 
 
 
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
         receivedStop_=false;
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
       bool receivedStop_;
       sem_t *control_sem_;
       pthread_mutex_t * mst_lock_;
       void * fuAddr;
   };
 }

}
#endif
