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

namespace cscdqm {

  typedef boost::recursive_mutex::scoped_lock LockType;

  /**
   * @class Lock
   * @brief Lockable interface that blocks thread
   */
  class Lock {
  
    public:

      /** Mutual exclusion object */
      boost::recursive_mutex mutex;

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
