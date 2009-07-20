// $Id: StateMachine.h,v 1.7 2009/07/14 10:34:44 dshpakov Exp $
/// @file: RegistrationQueue.h 

#ifndef EventFilter_StorageManager_RegistrationQueue_h
#define EventFilter_StorageManager_RegistrationQueue_h

#include <boost/shared_ptr.hpp>
#include <EventFilter/StorageManager/interface/RegistrationInfoBase.h>

namespace stor
{

  /**
     Concurrent queue holding consumer registrations

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */
  typedef boost::shared_ptr<RegistrationInfoBase> RegInfoBasePtr;
  typedef ConcurrentQueue<RegInfoBasePtr> RegistrationQueue;
}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
