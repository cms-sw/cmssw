// $Id: StorageManager.h,v 1.51 2009/07/10 11:41:03 dshpakov Exp $
/// @file: StorageManager.h 

#ifndef StorageManager_StorageManager_h
#define StorageManager_StorageManager_h

#include <string>

#include "EventFilter/StorageManager/interface/DiskWriter.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessor.h"
#include "EventFilter/StorageManager/interface/FragmentProcessor.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/WebPageHelper.h"

#include "xdaq/Application.h"
#include "xgi/exception/Exception.h"
#include "xoap/MessageReference.h"


namespace toolbox { 
  namespace mem {
    class Reference;
  }
}

namespace xgi {
  class Input;
  class Output;
}

namespace stor {

  /**
   * Main class of the StorageManager XDAQ application
   *
   * $Author: dshpakov $
   * $Revision: 1.51 $
   * $Date: 2009/07/10 11:41:03 $
   */

  class StorageManager: public xdaq::Application
  {

  public:
  
    StorageManager( xdaq::ApplicationStub* s );
  
    ~StorageManager();


  private:  
  
    StorageManager(StorageManager const&); // not implemented
    StorageManager& operator=(StorageManager const&); // not implemented

    /**
     * Bind callbacks for I2O message
     */
    void bindI2OCallbacks();

    /**
     * Callback for I2O message containing an init message
     */
    void receiveRegistryMessage(toolbox::mem::Reference *ref);

    /**
     * Callback for I2O message containing an event
     */
    void receiveDataMessage(toolbox::mem::Reference *ref);

    /**
     * Callback for I2O message containing an error event
     */
    void receiveErrorDataMessage(toolbox::mem::Reference *ref);

    /**
     * Callback for I2O message containing a DQM event (histogramms)
     */
    void receiveDQMMessage(toolbox::mem::Reference *ref);

    /**
     * Bind callbacks for state machine SOAP messages
     */
    void bindStateMachineCallbacks();

    /**
     * Callback for SOAP message containint a state machine event,
     * possibly including new configuration values
     */
    xoap::MessageReference handleFSMSoapMessage( xoap::MessageReference )
      throw( xoap::exception::Exception );


    /**
     * Bind callbacks for web interface
     */
    void bindWebInterfaceCallbacks();

    /**
     * Webinterface callback for style sheet
     */
    void css(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating default web page
     */
    void defaultWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing the stored data information
     */
    void storedDataWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing information about
     * recently written files
     */
    void fileStatisticsWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing summary information
     * about the resource broker sending data.
     */
    void rbsenderWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing detailed information
     * about the resource broker sending data.
     */
    void rbsenderDetailWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing the connected consumers
     */
    void consumerStatisticsPage( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );

    /**
     * Webinterface callback creating web page showing statistics about the
     * processed DQM events.
     */
    void dqmEventStatisticsWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Webinterface callback creating web page showing statistics about the
     * data throughput in the SM.
     */
    void throughputWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Callback returning a XML list of consumer information.
     * The current implementation just returns an empty document.
     */
    void consumerListWebPage(xgi::Input *in, xgi::Output *out)
      throw (xgi::exception::Exception);

    /**
     * Bind callbacks for consumers
     */
    void bindConsumerCallbacks();

    /**
     * Callback handling event consumer registration request
     */
    void processConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );

    /**
     * Callback handling event consumer init message request
     */
    void processConsumerHeaderRequest( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );

    /**
     * Callback handling event consumer event request
     */
    void processConsumerEventRequest( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );
 
    /**
     * Callback handling DQM event consumer registration request
     */
    void processDQMConsumerRegistrationRequest( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );

    /**
     * Callback handling DQM event consumer DQM event request
     */
    void processDQMConsumerEventRequest( xgi::Input* in, xgi::Output* out )
      throw( xgi::exception::Exception );

    /**
     * Initialize the shared resources
     */
    void initializeSharedResources();

    /**
     * Create and start all worker threads
     */
    void startWorkerThreads();

    // instantiate the plugin manager, not referenced here after!
    edm::AssertHandler _ah;

    SharedResourcesPtr _sharedResources;

    FragmentProcessor *_fragmentProcessor;
    DiskWriter *_diskWriter;
    DQMEventProcessor *_dqmEventProcessor;

    WebPageHelper _webPageHelper;

  };

}

#endif // StorageManager_StorageManager_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
