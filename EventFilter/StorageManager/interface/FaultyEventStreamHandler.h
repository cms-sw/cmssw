// $Id: FaultyEventStreamHandler.h,v 1.6 2010/03/19 13:24:30 mommsen Exp $
/// @file: FaultyEventStreamHandler.h 

#ifndef StorageManager_FaultyEventStreamHandler_h
#define StorageManager_FaultyEventStreamHandler_h

#include <string>

#include "IOPool/Streamer/interface/InitMessage.h"

#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {


  /**
   * Handle the faulty event stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2010/03/19 13:24:30 $
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
    { return _streamRecord->streamName; }

    /**
     * Return the fraction-to-disk parameter
     */
    virtual double fractionToDisk() const
    { return _streamRecord->fractionToDisk; }

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


    boost::shared_ptr<InitMsgCollection> _initMsgCollection;

  };
  
} // namespace stor

#endif // StorageManager_FaultyEventStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
