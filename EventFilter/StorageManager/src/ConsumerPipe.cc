/**
 * This class is used to manage the subscriptions, events, and
 * lost connections associated with an event consumer within the
 * event server part of the storage manager.
 *
 * 16-Aug-2006 - KAB  - Initial Implementation
 */

#include "EventFilter/StorageManager/interface/ConsumerPipe.h"

using namespace std;
using namespace stor;

/**
 * Initialize the static value for the root consumer id.
 */
uint32 ConsumerPipe::rootId_ = 1;

/**
 * Initialize the static lock used to control access to the root ID.
 */
boost::mutex ConsumerPipe::rootIdLock_;

/**
 * ConsumerPipe constructor.
 */
ConsumerPipe::ConsumerPipe(std::string name, std::string priority):
  consumerName_(name),consumerPriority_(priority)
{
  boost::mutex::scoped_lock scopedLockForRootId(rootIdLock_);
  consumerId_ = rootId_;
  rootId_++;
}

/**
 * ConsumerPipe destructor.
 */
ConsumerPipe::~ConsumerPipe()
{
}

/**
 * Returns the consumer ID associated with this pipe.
 */
uint32 ConsumerPipe::getConsumerId()
{
  return consumerId_;
}
