// $Id: EventStreamHandler.h,v 1.4 2009/08/28 16:41:49 mommsen Exp $
/// @file: EventStreamHandler.h 

#ifndef StorageManager_EventStreamHandler_h
#define StorageManager_EventStreamHandler_h

#include <string>

#include "IOPool/Streamer/interface/InitMessage.h"

#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {

  class Configuration;
  class EventStreamConfigurationInfo;


  /**
   * Handle one event stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2009/08/28 16:41:49 $
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
