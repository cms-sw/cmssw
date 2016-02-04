// $Id: DQMEventProcessorResources.cc,v 1.5 2011/03/07 15:31:32 mommsen Exp $
/// @file: DQMEventProcessorResources.cc

#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"

namespace stor
{
  DQMEventProcessorResources::DQMEventProcessorResources() :
  requestsPending_(false),
  requestsInProgress_(false)
  {
    pendingRequests_.reset();
  }
  
  void DQMEventProcessorResources::
  requestConfiguration(DQMProcessingParams const& params, boost::posix_time::time_duration const& timeoutValue)
  {
    boost::mutex::scoped_lock sl(requestsMutex_);

    requestedDQMProcessingParams_ = params;
    requestedTimeout_ = timeoutValue;

    // A new configuration forces the store destruction and
    // after the store is destroyed, end-of-run processing
    // has nothing left to do. Thus, cancel these requests.
    pendingRequests_.configuration = true;
    pendingRequests_.endOfRun = false;
    pendingRequests_.storeDestruction = false;
    requestsPending_ = true;
  }

  void DQMEventProcessorResources::requestEndOfRun()
  {
    boost::mutex::scoped_lock sl(requestsMutex_);

    // A end-of-run request does not change any other requests.
    pendingRequests_.endOfRun = true;
    requestsPending_ = true;
  }

  void DQMEventProcessorResources::requestStoreDestruction()
  {
    boost::mutex::scoped_lock sl(requestsMutex_);

    // The store destruction clears everything.
    // Thus, cancel any other pending requests.
    pendingRequests_.configuration = false;
    pendingRequests_.endOfRun = false;
    pendingRequests_.storeDestruction = true;
    requestsPending_ = true;
  }

  bool DQMEventProcessorResources::
  getRequests(Requests& requests, DQMProcessingParams& params, boost::posix_time::time_duration& timeoutValue)
  {
    boost::mutex::scoped_lock sl(requestsMutex_);

    if (! requestsPending_) {return false;}

    requestsPending_ = false;

    requests = pendingRequests_;
    params = requestedDQMProcessingParams_;
    timeoutValue = requestedTimeout_;

    pendingRequests_.reset();
    requestsInProgress_ = true;

    return true;
  }

  void DQMEventProcessorResources::waitForCompletion()
  {
    boost::mutex::scoped_lock sl(requestsMutex_);
    if (requestsPending_ || requestsInProgress_)
      {
        requestsCondition_.wait(sl);
      }
  }

  bool DQMEventProcessorResources::requestsOngoing()
  {
    boost::mutex::scoped_lock sl(requestsMutex_);
    return (requestsPending_ || requestsInProgress_);
  }

  void DQMEventProcessorResources::requestsDone()
  {
    boost::mutex::scoped_lock sl(requestsMutex_);
    if (requestsInProgress_)
      {
        requestsCondition_.notify_one();
      }
    requestsInProgress_ = false;
  }

  void DQMEventProcessorResources::Requests::reset()
  {
    configuration = false;
    endOfRun = false;
    storeDestruction = false;
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
