// $Id: DQMEventProcessorResources.h,v 1.5 2011/03/07 15:31:31 mommsen Exp $
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
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2011/03/07 15:31:31 $
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
    void requestConfiguration(DQMProcessingParams const&,  boost::posix_time::time_duration const&);

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
    bool getRequests(Requests&, DQMProcessingParams&, boost::posix_time::time_duration& timeoutValue);

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

    bool requestsPending_;
    bool requestsInProgress_;
    Requests pendingRequests_;

    DQMProcessingParams requestedDQMProcessingParams_;
    boost::posix_time::time_duration requestedTimeout_;

    boost::condition requestsCondition_;

    mutable boost::mutex requestsMutex_;
  };

  typedef boost::shared_ptr<DQMEventProcessorResources> DQMEventProcessorResourcesPtr;

} // namespace stor

#endif // EventFilter_StorageManager_DQMEventProcessorResources_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
