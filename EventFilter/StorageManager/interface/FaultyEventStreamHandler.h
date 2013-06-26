// $Id: FaultyEventStreamHandler.h,v 1.2 2011/03/07 15:31:31 mommsen Exp $
/// @file: FaultyEventStreamHandler.h 

#ifndef EventFilter_StorageManager_FaultyEventStreamHandler_h
#define EventFilter_StorageManager_FaultyEventStreamHandler_h

#include <string>

#include "IOPool/Streamer/interface/InitMessage.h"

#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {


  /**
   * Handle the faulty event stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:31:31 $
   */
  
  class FaultyEventStreamHandler : public StreamHandler
  {
  public:
    
    FaultyEventStreamHandler
    (
      const SharedResourcesPtr,
      const DbFileHandlerPtr,
      const std::string& streamName
    );


  private:

    /**
     * Return the stream label
     */
    virtual std::string streamLabel() const
    { return streamRecord_->streamName; }

    /**
     * Return the fraction-to-disk parameter
     */
    virtual double fractionToDisk() const
    { return streamRecord_->fractionToDisk; }

    /**
     * Get the file handler responsible for the event
     */    
    virtual FileHandlerPtr getFileHandler(const I2OChain& event);

    /**
     *  Return a new file handler for the provided event
     */    
    virtual FileHandlerPtr newFileHandler(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual int getStreamMaxFileSize() const
    { return 0; }


    InitMsgCollectionPtr initMsgCollection_;

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FaultyEventStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
