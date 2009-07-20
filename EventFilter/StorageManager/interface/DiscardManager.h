// $Id: DiscardManager.h,v 1.3 2009/06/24 19:11:21 biery Exp $
/// @file: DiscardManager.h 

#ifndef StorageManager_DiscardManager_h
#define StorageManager_DiscardManager_h

#include "EventFilter/StorageManager/interface/DataSenderMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FUProxy.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"

#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationDescriptor.h"
#include "toolbox/mem/Pool.h"

#include "boost/shared_ptr.hpp"

#include <map>

namespace stor {

  /**
   * Handles the discard messages sent to the upstream Resource Brokers.
   *
   * $Author: biery $
   * $Revision: 1.3 $
   * $Date: 2009/06/24 19:11:21 $
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
    DiscardManager(xdaq::ApplicationContext* ctx,
                   xdaq::ApplicationDescriptor* desc,
                   DataSenderMonitorCollection& dsmc);

    ~DiscardManager() {}

    /**
     * Configures the discard manager.  Internally, this connects
     * the DiscardManager to the buffer pool that it will use to store
     * the discard messages.  This should be done at configuration time.
     * (Application startup is too soon because the memory pool for
     * TCP messages may not have been created yet.  Also, it's probably
     * possible, in principle, for the configuration of resource brokers
     * to change while the SM is in the Halted state.)
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
    bool sendDiscardMessage(I2OChain const& i2oMessage);

  private:

    FUProxyPtr getProxyFromCache(std::string hltClassName,
                                 unsigned int hltInstance);
    FUProxyPtr makeNewFUProxy(std::string hltClassName,
                              unsigned int hltInstance);

    xdaq::ApplicationContext* _appContext;
    xdaq::ApplicationDescriptor* _appDescriptor;
    toolbox::mem::Pool* _pool;

    FUProxyMap _proxyCache;

    DataSenderMonitorCollection& _dataSenderMonCollection;
  };

} // namespace stor

#endif // StorageManager_DiscardManager_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
