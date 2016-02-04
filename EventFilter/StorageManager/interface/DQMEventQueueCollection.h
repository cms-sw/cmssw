// $Id: DQMEventQueueCollection.h,v 1.3 2009/07/20 13:06:10 mommsen Exp $
/// @file: DQMEventQueueCollection.h 

#ifndef StorageManager_DQMEventQueueCollection_h
#define StorageManager_DQMEventQueueCollection_h

#include "EventFilter/StorageManager/interface/DQMEventRecord.h"
#include "EventFilter/StorageManager/interface/QueueCollection.h"

namespace stor {

  /**
   * A collection of ConcurrentQueue<DQMEventRecord>.
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/20 13:06:10 $
   */

  typedef QueueCollection<DQMEventRecord::GroupRecord> DQMEventQueueCollection;
  
} // namespace stor

#endif // StorageManager_DQMEventQueueCollection_h 



// emacs configuration
// Local Variables: -
// mode: c++ -
// c-basic-offset: 2 -
// indent-tabs-mode: nil -
// End: -
