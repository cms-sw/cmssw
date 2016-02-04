// $Id: DiscardManager.h,v 1.7 2011/04/07 08:02:03 mommsen Exp $
/// @file: DiscardManager.h 

#ifndef EventFilter_StorageManager_DiscardManager_h
#define EventFilter_StorageManager_DiscardManager_h

#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationDescriptor.h"
#include "toolbox/mem/Pool.h"

#include "boost/shared_ptr.hpp"

#include <map>
#include <string>
#include <utility>


namespace stor {

  class DataSenderMonitorCollection;
  class FUProxy;
  class I2OChain;


  /**
   * Handles the discard messages sent to the upstream Resource Brokers.
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2011/04/07 08:02:03 $
   */

  class DiscardManager
  {
  public:

    typedef std::pair<std::string, unsigned int> HLTSenderKey;
    typedef boost::shared_ptr<FUProxy> FUProxyPtr;
    typedef std::map< HLTSenderKey, boost::shared_ptr<FUProxy> > FUProxyMap;

    /**
     * Creates a DiscardManager that will send discard messages
     * to upstream Resource Brokers on behalf of the application
     * specified in the application descriptor.  The DiscardManager
     * will use the specified application context to send the messages.
     */
    DiscardManager
    (
      xdaq::ApplicationContext*,
      xdaq::ApplicationDescriptor*,
      DataSenderMonitorCollection&
    );

    /**
     * Configures the discard manager.
     */
    void configure();

    /**
     * Sends a message to the appropriate resource broker
     * telling it that the SM has received and processed the
     * specified I2O message.  At that point, we expect that the
     * resource broker will free up the buffer that contained the
     * original event (or INIT message or DQMEvent or whatever)
     * and do whatever other cleanup it may need to do.
     *
     * There are two failure modes to this method.  In the first,
     * the I2OChain could be so badly corrupt that the target
     * resource broker can not be determined.  In that case, this
     * method simply returns false.  (Since we expect that these
     * messages will be sent to a special error stream, we will
     * rely on their presence in that stream to indicate that 
     * something went very wrong.)  In the second failure mode,
     * the I2OChain was parsable, but the lookup of the RB in
     * the XDAQ network failed.  In that case, this method throws
     * an exception.  (This should never happen, so we will treat
     * it as an exceptional condition.)
     *
     * @throws a stor::exception::RBLookupFailed exception if
     *         the appropriate resource broker can not be 
     *         determined.
     */
    bool sendDiscardMessage(I2OChain const&);

  private:

    FUProxyPtr getProxyFromCache
    (
      std::string const& hltClassName,
      unsigned int const& hltInstance
    );
    
    FUProxyPtr makeNewFUProxy
    (
      std::string const& hltClassName,
      unsigned int const& hltInstance
    );

    xdaq::ApplicationContext* appContext_;
    xdaq::ApplicationDescriptor* appDescriptor_;
    toolbox::mem::Pool* msgPool_;

    FUProxyMap proxyCache_;

    DataSenderMonitorCollection& dataSenderMonCollection_;
  };

  typedef boost::shared_ptr<DiscardManager> DiscardManagerPtr;

} // namespace stor

#endif // EventFilter_StorageManager_DiscardManager_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
