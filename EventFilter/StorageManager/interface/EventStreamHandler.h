// $Id: EventStreamHandler.h,v 1.6.8.2 2011/02/28 17:56:15 mommsen Exp $
/// @file: EventStreamHandler.h 

#ifndef EventFilter_StorageManager_EventStreamHandler_h
#define EventFilter_StorageManager_EventStreamHandler_h

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
   * $Revision: 1.6.8.2 $
   * $Date: 2011/02/28 17:56:15 $
   */
  
  class EventStreamHandler : public StreamHandler
  {
  public:
    
    EventStreamHandler
    (
      const EventStreamConfigurationInfo&,
      const SharedResourcesPtr,
      const DbFileHandlerPtr
    );


  private:

    /**
     * Return the stream label
     */
    virtual std::string streamLabel() const
    { return streamConfig_.streamLabel(); }

    /**
     * Return the fraction-to-disk parameter
     */
    virtual double fractionToDisk() const
    { return streamConfig_.fractionToDisk(); }

    /**
     * Return a new file handler for the provided event
     */    
    virtual FileHandlerPtr newFileHandler(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual int getStreamMaxFileSize() const
    { return streamConfig_.maxFileSizeMB(); }


    EventStreamConfigurationInfo streamConfig_;
    InitMsgCollectionPtr initMsgCollection_;
    InitMsgSharedPtr initMsgView_;

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_EventStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
