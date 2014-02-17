// $Id: DiskWriterResources.h,v 1.9 2011/03/07 15:31:31 mommsen Exp $
/// @file: DiskWriterResources.h 


#ifndef EventFilter_StorageManager_DiskWriterResources_h
#define EventFilter_StorageManager_DiskWriterResources_h

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"

#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/thread/condition.hpp"
#include "boost/thread/mutex.hpp"

#include <deque>
#include <stdint.h>


namespace stor
{

  /**
   * Container class for resources that are needed by the DiskWriter
   * and need to be accessed from multiple threads.
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class DiskWriterResources
  {
  public:

    /**
     * Constructs a DiskWriterResources instance.
     */
    DiskWriterResources();

    /**
     * Destructor.
     */
    virtual ~DiskWriterResources() {}

    /**
     * Requests that the DiskWriter streams be configured with the
     * specified configurations.  Also allows new DiskWritingParams,
     * run number and dequeue timeout values to be specified.
     * Existing stream configurations will be discarded. 
     */
    void requestStreamConfiguration
    (
      EvtStrConfigListPtr const,
      ErrStrConfigListPtr const,
      DiskWritingParams const&,
      unsigned int const& runNumber,
      boost::posix_time::time_duration const& timeoutValue
    );

    /**
     * Requests that the DiskWriter streams be destroyed.
     * Any pending stream configuraton request will be cancelled.
     */
    void requestStreamDestruction();

    /**
     * Checks if a request has been made to change the stream configuration
     * in the DiskWriter streams *and* clears any pending request.
     * Existing streams are purged.
     * If doConfig is true, the supplied new configurations, new run number, and
     * dequeue timeout value should be used to configure a new set of DiskWriter streams.
     */
    bool streamChangeRequested
    (
      bool& doConfig,
      EvtStrConfigListPtr&,
      ErrStrConfigListPtr&,
      DiskWritingParams& dwParams,
      unsigned int& runNumber,
      boost::posix_time::time_duration& timeoutValue
    );

    /**
     * Waits until a requested stream configuration change has been completed.
     */
    virtual void waitForStreamChange();

    /**
     * Returns true when a stream configuration change is requested or in progress.
     */
    virtual bool streamChangeOngoing();

    /**
     * Indicates that the stream configuration change is done.
     */
    void streamChangeDone();

    /**
     * Sets the DiskWriter "busy" status to the specified state.
     */
    void setBusy(bool isBusyFlag);

    /**
     * Tests if the DiskWriter is currently busy..
     */
    bool isBusy();

  private:

    bool configurationIsNeeded_;
    bool streamChangeIsNeeded_;
    bool fileClosingTestIsNeeded_;
    bool diskWriterIsBusy_;

    EvtStrConfigListPtr requestedEventStreamConfig_;
    ErrStrConfigListPtr requestedErrorStreamConfig_;
    DiskWritingParams requestedDiskWritingParams_;
    unsigned int requestedRunNumber_;
    boost::posix_time::time_duration requestedTimeout_;

    bool streamChangeInProgress_;
    boost::condition streamChangeCondition_;
    
    mutable boost::mutex streamChangeMutex_;
  };
  
  typedef boost::shared_ptr<DiskWriterResources> DiskWriterResourcesPtr;

} // namespace stor

#endif // EventFilter_StorageManager_DiskWriterResources_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
