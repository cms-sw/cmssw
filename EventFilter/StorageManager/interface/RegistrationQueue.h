// $Id: RegistrationQueue.h,v 1.6 2011/03/07 15:31:32 mommsen Exp $
/// @file: RegistrationQueue.h 

#ifndef EventFilter_StorageManager_RegistrationQueue_h
#define EventFilter_StorageManager_RegistrationQueue_h

#include <boost/shared_ptr.hpp>
#include <EventFilter/StorageManager/interface/ConcurrentQueue.h>
#include <EventFilter/StorageManager/interface/RegistrationInfoBase.h>

namespace stor
{

  /**
     Concurrent queue holding consumer registrations

     $Author: mommsen $
     $Revision: 1.6 $
     $Date: 2011/03/07 15:31:32 $
  */
  typedef ConcurrentQueue<RegPtr> RegistrationQueue;
  typedef boost::shared_ptr<RegistrationQueue> RegistrationQueuePtr;

} // namespace stor

#endif // EventFilter_StorageManager_RegistrationQueue_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
