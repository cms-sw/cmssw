/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Lock.h
 *
 *    Description:  Lockable interface that blocks current thread.
 *
 *        Version:  1.0
 *        Created:  10/06/2008 01:49:51 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Lock_H
#define CSCDQM_Lock_H

#include <boost/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace cscdqm {

  /**
   * @class Lock
   * @brief Lockable interface that blocks thread
   */
  class Lock {
  
    private:

      /** Mutex object */
      boost::recursive_mutex lckMutex;

      /** Lock object */
      boost::recursive_mutex::scoped_lock lckLock;
     
      /** Flag to mark a locked state */
      bool lockedByOther; 

    public: 

      /**
       * @brief  Constructor.
       */
      Lock() : lckLock(lckMutex) { 
        unlock(); 
        lockedByOther = false;
      }

      /**
       * @brief  Destructor.
       */
      virtual ~Lock() { }

      /**
       * @brief  Lock object.
       */
      void lock() { 
        if (!isLocked()) {
          lckLock.lock(); 
          lockedByOther = true;
        }
      }

      /**
       * @brief  Unlock object.
       */
      void unlock() { 
        if (isLocked()) {
          lckLock.unlock(); 
          lockedByOther = false;
        }
      }

      /**
       * @brief  If I (this thread) have locked this object?
       * @return true if the object have locked this object, false - otherwise.
       */
      const bool isLocked() const { 
        return lckLock.locked(); 
      }

      /**
       * @brief  If someone else (another thread) have locked this object?
       * @return true if the object have locked another thread, false - otherwise.
       */
      const bool isLockedByOther() const {
        return lockedByOther;
      }

  };

}

#endif
