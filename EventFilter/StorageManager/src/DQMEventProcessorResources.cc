// $Id: DQMEventProcessorResources.cc,v 1.2 2009/06/10 08:15:25 dshpakov Exp $
/// @file: DQMEventProcessorResources.cc

#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"

namespace stor
{
  DQMEventProcessorResources::DQMEventProcessorResources() :
  _requestsPending(false),
  _requestsInProgress(false)
  {
    _pendingRequests.reset();
  }
  
  void DQMEventProcessorResources::
  requestConfiguration(DQMProcessingParams params, double timeoutValue)
  {
    boost::mutex::scoped_lock sl(_requestsMutex);

    _requestedDQMProcessingParams = params;
    _requestedTimeout = timeoutValue;

    // A new configuration forces the store destruction and
    // after the store is destroyed, end-of-run processing
    // has nothing left to do. Thus, cancel these requests.
    _pendingRequests.configuration = true;
    _pendingRequests.endOfRun = false;
    _pendingRequests.storeDestruction = false;
    _requestsPending = true;
  }

  void DQMEventProcessorResources::requestEndOfRun()
  {
    boost::mutex::scoped_lock sl(_requestsMutex);

    // A end-of-run request does not change any other requests.
    _pendingRequests.endOfRun = true;
    _requestsPending = true;
  }

  void DQMEventProcessorResources::requestStoreDestruction()
  {
    boost::mutex::scoped_lock sl(_requestsMutex);

    // The store destruction clears everything.
    // Thus, cancel any other pending requests.
    _pendingRequests.configuration = false;
    _pendingRequests.endOfRun = false;
    _pendingRequests.storeDestruction = true;
    _requestsPending = true;
  }

  bool DQMEventProcessorResources::
  getRequests(Requests& requests, DQMProcessingParams& params, double& timeoutValue)
  {
    boost::mutex::scoped_lock sl(_requestsMutex);

    if (! _requestsPending) {return false;}

    _requestsPending = false;

    requests = _pendingRequests;
    params = _requestedDQMProcessingParams;
    timeoutValue = _requestedTimeout;

    _pendingRequests.reset();
    _requestsInProgress = true;

    return true;
  }

  void DQMEventProcessorResources::waitForCompletion()
  {
    boost::mutex::scoped_lock sl(_requestsMutex);
    if (_requestsPending || _requestsInProgress)
      {
        _requestsCondition.wait(sl);
      }
  }

  bool DQMEventProcessorResources::requestsOngoing()
  {
    boost::mutex::scoped_lock sl(_requestsMutex);
    return (_requestsPending || _requestsInProgress);
  }

  void DQMEventProcessorResources::requestsDone()
  {
    boost::mutex::scoped_lock sl(_requestsMutex);
    if (_requestsInProgress)
      {
        _requestsCondition.notify_one();
      }
    _requestsInProgress = false;
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
