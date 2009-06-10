// $Id$

#ifndef StorageManager_FRDStreamHandler_h
#define StorageManager_FRDStreamHandler_h

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"


namespace stor {

  /**
   * Handle one FED Raw Data (error) event stream written to disk.
   *
   * $Author$
   * $Revision$
   * $Date$
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
