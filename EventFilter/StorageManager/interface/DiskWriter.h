// $Id: DiskWriter.h,v 1.11 2010/05/11 17:58:01 mommsen Exp $
/// @file: DiskWriter.h 

#ifndef StorageManager_DiskWriter_h
#define StorageManager_DiskWriter_h

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
   * $Revision: 1.11 $
   * $Date: 2010/05/11 17:58:01 $
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
    void closeTimedOutFiles(const utils::time_point_t);

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
    void reportRemainingLumiSections() const;

    /**
     * Log end-of-run marker
     */
    void writeEndOfRunMarker() const;


    xdaq::Application* _app;
    SharedResourcesPtr _sharedResources;
    DiskWritingParams _dwParams;
    const DbFileHandlerPtr _dbFileHandler;

    unsigned int _runNumber;
    boost::posix_time::time_duration _timeout; // Timeout on stream queue
    utils::time_point_t _lastFileTimeoutCheckTime; // Last time we checked for time-out files

    typedef boost::shared_ptr<StreamHandler> StreamHandlerPtr;
    typedef std::vector<StreamHandlerPtr> StreamHandlers;
    StreamHandlers _streamHandlers;

    bool _actionIsActive;
    toolbox::task::WorkLoop* _writingWL;      

  };
  
} // namespace stor

#endif // StorageManager_DiskWriter_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
