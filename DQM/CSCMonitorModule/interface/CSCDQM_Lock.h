/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Lock.h
 *
 *    Description:  Monitor Object interface
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

      boost::recursive_mutex lckMutex;
      boost::recursive_mutex::scoped_lock lckLock;
      bool lockedByOther; 

    public: 

      Lock() : lckLock(lckMutex) { 
        unlock(); 
        lockedByOther = false;
      }

      virtual ~Lock() { }

      void lock() { 
        if (!isLocked()) {
          lckLock.lock(); 
          lockedByOther = true;
        }
      }

      void unlock() { 
        if (isLocked()) {
          lckLock.unlock(); 
          lockedByOther = false;
        }
      }

      const bool isLocked() const { 
        return lckLock.locked(); 
        return false;
      }

      const bool isLockedByOther() const {
        return lockedByOther;
        return false;
      }

  };

}

#endif
