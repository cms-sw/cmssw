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

#ifdef DQMMT      
#include <boost/thread.hpp>
#endif

namespace cscdqm {

#ifdef DQMMT
  typedef boost::recursive_mutex::scoped_lock LockType;
#else
  struct LockType {
    bool locked;
    LockType(bool locked_) : locked(locked_) { }
    void unlock() { locked = false;  }
  };
#endif

  /**
   * @class Lock
   * @brief Lockable interface that blocks thread
   */
  class Lock {
  
    public:

      /** Mutual exclusion object */
#ifdef DQMMT
      boost::recursive_mutex mutex;
#else
      bool mutex;
#endif

      /**
       * @brief  Constructor.
       */
      Lock() { }

      /**
       * @brief  Destructor.
       */
      virtual ~Lock() { }

  };

}

#endif
