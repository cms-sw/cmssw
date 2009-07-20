// $Id: DiskWriter.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DiskWriter.h 

#ifndef StorageManager_DiskWriter_h
#define StorageManager_DiskWriter_h

#include "boost/shared_ptr.hpp"

#include <vector>

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {

  /**
   * Writes events to disk
   *
   * It gets the next event from the StreamQueue and writes it
   * to the appropriate stream file(s) on disk. 
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
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
     * Close all timed-out files
     */    
    void closeTimedOutFiles();

    /**
     * Returns true if the next check for timed-out files is due
     */    
    bool timeToCheckForFileTimeOut();

    /**
     * Configures the event streams to be written to disk
     */    
    void configureEventStreams(EvtStrConfigListPtr);

    /**
     * Configures the error streams to be written to disk
     */    
    void configureErrorStreams(ErrStrConfigListPtr);

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

    xdaq::Application* _app;
    SharedResourcesPtr _sharedResources;

    unsigned int _timeout; // Timeout in seconds on stream queue
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
