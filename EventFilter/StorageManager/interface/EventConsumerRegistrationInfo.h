// $Id: EventConsumerRegistrationInfo.h,v 1.4 2009/09/16 09:53:23 dshpakov Exp $
/// @file: EventConsumerRegistrationInfo.h 

#ifndef StorageManager_EventConsumerRegistrationInfo_h
#define StorageManager_EventConsumerRegistrationInfo_h

#include <iosfwd>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/CommonRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor
{
  /**
   * Holds the registration information from a event consumer.
   *
   * $Author: dshpakov $
   * $Revision: 1.4 $
   * $Date: 2009/09/16 09:53:23 $
   */

  class EventConsumerRegistrationInfo: public RegistrationInfoBase
  {

  public:

    typedef std::vector<std::string> FilterList;

    /**
     * Constructs an instance with the specified registration information.
     */
    EventConsumerRegistrationInfo( const unsigned int& maxConnectRetries,
				   const unsigned int& connectRetryInterval,// seconds
				   const std::string& consumerName,
				   const FilterList& selEvents,
				   const std::string& outputModuleLabel,
                                   const size_t& queueSize,
                                   const enquing_policy::PolicyTag& queuePolicy,
				   const utils::duration_t& secondsToStale,
                                   const std::string& remoteHost );

    ~EventConsumerRegistrationInfo();

    // Accessors:
    unsigned int maxConnectRetries() const { return _maxConnectRetries; }
    unsigned int connectRetryInterval() const { return _connectRetryInterval; }
    const FilterList& selEvents() const { return _selEvents; }
    const std::string& outputModuleLabel() const { return _outputModuleLabel; }
    bool isProxyServer() const { return _isProxy; }
    const std::string& remoteHost() const { return _remoteHost; }

    // Staleness:
    bool isStale() const { return _stale; }
    void setStaleness( bool s ) { _stale = s; }

    // Output:
    std::ostream& write(std::ostream& os) const;

    // Implementation of Template Method pattern.
    virtual void do_registerMe(EventDistributor*);
    virtual QueueID do_queueId() const;
    virtual void do_setQueueID(QueueID const& id);
    virtual std::string do_consumerName() const;
    virtual ConsumerID do_consumerId() const;
    virtual void do_setConsumerID(ConsumerID const& id);
    virtual size_t do_queueSize() const;
    virtual enquing_policy::PolicyTag do_queuePolicy() const;
    virtual utils::duration_t do_secondsToStale() const;

  private:

    CommonRegistrationInfo _common;

    unsigned int _maxConnectRetries;
    unsigned int _connectRetryInterval;
    FilterList _selEvents;
    std::string _outputModuleLabel;
    bool _isProxy;
    bool _stale;
    std::string _remoteHost;

  };

  typedef boost::shared_ptr<stor::EventConsumerRegistrationInfo> ConsRegPtr;

  /**
     Print the given EventConsumerRegistrationInfo to the given
     stream.
  */
  inline
  std::ostream& operator << ( std::ostream& os, 
                              EventConsumerRegistrationInfo const& ri )
  {
    return ri.write( os );
  }

} // namespace stor

#endif // StorageManager_EventConsumerRegistrationInfo_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
