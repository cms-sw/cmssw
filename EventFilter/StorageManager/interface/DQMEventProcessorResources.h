// $Id: DQMEventProcessorResources.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DQMEventProcessorResources.h 


#ifndef EventFilter_StorageManager_DQMEventProcessorResources_h
#define EventFilter_StorageManager_DQMEventProcessorResources_h

#include "EventFilter/StorageManager/interface/Configuration.h"

#include "boost/thread/condition.hpp"
#include "boost/thread/mutex.hpp"

namespace stor
{

  /**
   * Container class for resources that are needed by the DQMEventProcessor
   * and need to be accessed from multiple threads.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */

  class DQMEventProcessorResources
  {
  public:

    struct Requests
    {
      bool configuration;
      bool endOfRun;
      bool storeDestruction;

      void reset();
    };

    /**
     * Constructs a DQMEventProcessorResources instance.
     */
    DQMEventProcessorResources();

    /**
     * Destructor.
     */
    virtual ~DQMEventProcessorResources() {}

    /**
     * Requests that the DQMEventProcessor be configured with the
     * specified DQMProcessingParams.  Also allows a new dequeue timeout
     * value to be specified.
     */
    void requestConfiguration(DQMProcessingParams, double timeoutValue);

    /**
     * Requests the end-of-run processing
     */
    void requestEndOfRun();

    /**
     * Requests that the DQMEventStore is emptied.
     */
    void requestStoreDestruction();

    /**
     * Checks if a request has been made *and* clears any pending request.
     * Supplies the new DQMProcessingParams and a new dequeue timeout value
     * if a new configuration is requested.
     */
    bool getRequests(Requests&, DQMProcessingParams&, double& timeoutValue);

    /**
     * Waits until the requests have been completed.
     */
    virtual void waitForCompletion();

    /**
     * Returns true when requests are pending or being processed
     */
    virtual bool requestsOngoing();

    /**
     * Indicates that the requests were processed
     */
    void requestsDone();


  private:

    bool _requestsPending;
    bool _requestsInProgress;
    Requests _pendingRequests;

    DQMProcessingParams _requestedDQMProcessingParams;
    double _requestedTimeout;

    boost::condition _requestsCondition;

    mutable boost::mutex _requestsMutex;
  };

}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
