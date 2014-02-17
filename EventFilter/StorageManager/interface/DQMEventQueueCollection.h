// $Id: DQMEventQueueCollection.h,v 1.4 2011/03/07 15:31:31 mommsen Exp $
/// @file: DQMEventQueueCollection.h 

#ifndef EventFilter_StorageManager_DQMEventQueueCollection_h
#define EventFilter_StorageManager_DQMEventQueueCollection_h

#include "boost/shared_ptr.hpp"
#include "EventFilter/StorageManager/interface/DQMTopLevelFolder.h"
#include "EventFilter/StorageManager/interface/QueueCollection.h"

namespace stor {

  /**
   * A collection of ConcurrentQueue<DQMEventRecord>.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/03/07 15:31:31 $
   */

  typedef QueueCollection<DQMTopLevelFolder::Record> DQMEventQueueCollection;
  typedef boost::shared_ptr<DQMEventQueueCollection> DQMEventQueueCollectionPtr;

} // namespace stor

#endif // EventFilter_StorageManager_DQMEventQueueCollection_h 



// emacs configuration
// Local Variables: -
// mode: c++ -
// c-basic-offset: 2 -
// indent-tabs-mode: nil -
// End: -
