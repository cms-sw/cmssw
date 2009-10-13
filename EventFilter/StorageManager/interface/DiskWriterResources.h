// $Id: DiskWriterResources.h,v 1.4 2009/09/16 13:30:47 mommsen Exp $
/// @file: DiskWriterResources.h 


#ifndef EventFilter_StorageManager_DiskWriterResources_h
#define EventFilter_StorageManager_DiskWriterResources_h

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"

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
   * $Revision: 1.4 $
   * $Date: 2009/09/16 13:30:47 $
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
     * specified configurations.  Also allows a new dequeue timeout
     * value to be specified. Existing stream configurations will be
     * discarded. 
     */
    void requestStreamConfiguration
    (
      EvtStrConfigListPtr,
      ErrStrConfigListPtr,
      double timeoutValue
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
     * If doConfig is true, the supplied new configurations and a new dequeue
     * timeout value should be used to configure a new set of DiskWriter streams.
     */
    bool streamChangeRequested
    (
      bool& doConfig,
      EvtStrConfigListPtr&,
      ErrStrConfigListPtr&,
      double& timeoutValue
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
     * Requests that the DiskWriter closes all files for the 
     * specified lumi section.
     */
    void requestLumiSectionClosure(const uint32_t lumiSection);

    /**
     * Checks if a request has been made to close all files for
     * a lumi section. If it returns true, the argument contains
     * the lumi section number to be closed.
     */
    bool lumiSectionClosureRequested(uint32_t& lumiSection);

    /**
     * Sets the DiskWriter "busy" status to the specified state.
     */
    void setBusy(bool isBusyFlag);

    /**
     * Tests if the DiskWriter is currently busy..
     */
    bool isBusy();

  private:

    bool _configurationIsNeeded;
    bool _streamChangeIsNeeded;
    bool _fileClosingTestIsNeeded;
    bool _diskWriterIsBusy;

    EvtStrConfigListPtr _requestedEventStreamConfig;
    ErrStrConfigListPtr _requestedErrorStreamConfig;
    double _requestedTimeout;

    bool _streamChangeInProgress;
    boost::condition _streamChangeCondition;
    
    std::deque<uint32_t> lumiSectionsToClose;

    mutable boost::mutex _streamChangeMutex;
    mutable boost::mutex _lumiSectionMutex;
  };

}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
