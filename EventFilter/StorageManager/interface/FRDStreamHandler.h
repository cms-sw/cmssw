// $Id: FRDStreamHandler.h,v 1.4 2009/08/28 16:41:49 mommsen Exp $
/// @file: FRDStreamHandler.h 

#ifndef StorageManager_FRDStreamHandler_h
#define StorageManager_FRDStreamHandler_h

#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {

  class ErrorStreamConfigurationInfo;
  class I2OChain;


  /**
   * Handle one FED Raw Data (error) event stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2009/08/28 16:41:49 $
   */
  
  class FRDStreamHandler : public StreamHandler
  {
  public:
    
    FRDStreamHandler
    (
      const ErrorStreamConfigurationInfo&,
      SharedResourcesPtr
    );


  private:

    /**
     * Return the stream label
     */
    virtual std::string streamLabel() const
    { return _streamConfig.streamLabel(); }

    /**
     * Return the fraction-to-disk parameter
     * This value is not configurable for FRD events,
     * i.e. all events are always written
     */
    virtual double fractionToDisk() const
    { return 1; }

    /**
     * Return a new file handler for the provided event
     */    
    virtual FileHandlerPtr newFileHandler(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual int getStreamMaxFileSize() const
    { return _streamConfig.maxFileSizeMB(); }


    ErrorStreamConfigurationInfo _streamConfig;
    
  };
  
} // namespace stor

#endif // StorageManager_FRDStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
