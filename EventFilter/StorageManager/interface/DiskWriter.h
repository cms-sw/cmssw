// $Id: DiskWriter.h,v 1.15 2011/06/20 15:55:52 mommsen Exp $
/// @file: DiskWriter.h 

#ifndef EventFilter_StorageManager_DiskWriter_h
#define EventFilter_StorageManager_DiskWriter_h

#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/shared_ptr.hpp"

#include <stdint.h>
#include <vector>

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DbFileHandler.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {

  class I2OChain;
  class StreamHandler;

  /**
   * Writes events to disk
   *
   * It gets the next event from the StreamQueue and writes it
   * to the appropriate stream file(s) on disk. 
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2011/06/20 15:55:52 $
   */
  
  class DiskWriter : public toolbox::lang::Class
  {
  public:


    DiskWriter(xdaq::Application*, SharedResourcesPtr sr);

    ~DiskWriter();


    /**
     * The workloop action taking the next event from the StreamQueue
     * and writing it to disk
     */    
    bool writeAction(toolbox::task::WorkLoop*);

    /**
     * Creates and starts the disk writing workloop
     */
    void startWorkLoop(std::string workloopName);


  private:

    //Prevent copying of the DiskWriter
    DiskWriter(DiskWriter const&);
    DiskWriter& operator=(DiskWriter const&);


    /**
     * Takes the event from the stream queue
     */    
    void writeNextEvent();

    /**
     * Writes the event to the appropriate streams
     */    
    void writeEventToStreams(const I2OChain&);

    /**
     * Reconfigure streams if a request is pending
     */    
    void checkStreamChangeRequest();

    /**
     * Close old files if fileClosingTestInterval has passed
     * or do it now if argument is true
     */    
    void checkForFileTimeOuts(const bool doItNow = false);

    /**
     * Close all timed-out files
     */    
    void closeTimedOutFiles(const utils::TimePoint_t);

    /**
     * Configures the event streams to be written to disk
     */    
    void configureEventStreams(EvtStrConfigListPtr);

    /**
     * Configures the error streams to be written to disk
     */    
    void configureErrorStreams(ErrStrConfigListPtr);

    /**
     * Creates the handler for faulty events detected by the SM
     */    
    void makeFaultyEventStream();

    /**
     * Creates the handler for the given event stream
     */    
    void makeEventStream(EventStreamConfigurationInfo&);

    /**
     * Creates the handler for the given error event stream
     */    
    void makeErrorStream(ErrorStreamConfigurationInfo&);

    /**
     * Gracefully close all streams
     */    
    void destroyStreams();

    /**
     * Close files at the end of a luminosity section and release
     * message memory:
     */
    void processEndOfLumiSection(const I2OChain&);

    /**
     * Log file statistics for so far unreported lumi sections
     */
    void reportRemainingLumiSections();

    /**
     * Log end-of-run marker
     */
    void writeEndOfRunMarker();


    xdaq::Application* app_;
    SharedResourcesPtr sharedResources_;
    DiskWritingParams dwParams_;
    const DbFileHandlerPtr dbFileHandler_;

    unsigned int runNumber_;
    boost::posix_time::time_duration timeout_; // Timeout on stream queue
    utils::TimePoint_t lastFileTimeoutCheckTime_; // Last time we checked for time-out files

    typedef boost::shared_ptr<StreamHandler> StreamHandlerPtr;
    typedef std::vector<StreamHandlerPtr> StreamHandlers;
    StreamHandlers streamHandlers_;

    StreamsMonitorCollection::EndOfRunReportPtr endOfRunReport_;

    bool actionIsActive_;
    toolbox::task::WorkLoop* writingWL_;      

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_DiskWriter_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
