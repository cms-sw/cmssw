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
   * @brief Lockable interface 
   */
  class Lock {
  
    private:

      boost::recursive_mutex lckMutex;
      boost::recursive_mutex::scoped_lock lckLock;

    public: 

      Lock() : lckLock(lckMutex) { unlock(); }
      virtual ~Lock() { unlock(); }

      void lock()   { 
        if (!isLocked()) lckLock.lock(); 
      }

      void unlock() { 
        if (isLocked()) lckLock.unlock(); 
      }

      const bool isLocked() const { 
        return lckLock.locked(); 
      }
  };

}

#endif
