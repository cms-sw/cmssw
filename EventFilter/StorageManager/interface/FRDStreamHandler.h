// $Id: FRDStreamHandler.h,v 1.7 2011/03/07 15:31:31 mommsen Exp $
/// @file: FRDStreamHandler.h 

#ifndef EventFilter_StorageManager_FRDStreamHandler_h
#define EventFilter_StorageManager_FRDStreamHandler_h

#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {

  class ErrorStreamConfigurationInfo;
  class I2OChain;


  /**
   * Handle one FED Raw Data (error) event stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2011/03/07 15:31:31 $
   */
  
  class FRDStreamHandler : public StreamHandler
  {
  public:
    
    FRDStreamHandler
    (
      const ErrorStreamConfigurationInfo&,
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
    { return streamConfig_.maxFileSizeMB(); }


    ErrorStreamConfigurationInfo streamConfig_;
    
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FRDStreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
