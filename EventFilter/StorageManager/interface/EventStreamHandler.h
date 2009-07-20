// $Id: EventStreamHandler.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $
/// @file: EventStreamHandler.h 

#ifndef StorageManager_EventStreamHandler_h
#define StorageManager_EventStreamHandler_h

#include <string>

#include "IOPool/Streamer/interface/InitMessage.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {

  /**
   * Handle one event stream written to disk.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:22 $
   */
  
  class EventStreamHandler : public StreamHandler
  {
  public:
    
    EventStreamHandler
    (
      const EventStreamConfigurationInfo&,
      SharedResourcesPtr
    );


  private:

    /**
     * Return the stream label
     */
    virtual const std::string streamLabel() const
    { return _streamConfig.streamLabel(); }

    /**
     * Return a new file handler for the provided event
     */    
    virtual const FileHandlerPtr newFileHandler(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual const int getStreamMaxFileSize() const
    { return _streamConfig.maxFileSizeMB(); }


    EventStreamConfigurationInfo _streamConfig;
    boost::shared_ptr<InitMsgCollection> _initMsgCollection;
    InitMsgSharedPtr _initMsgView;

  };
  
} // namespace stor

#endif // StorageManager_EventStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
