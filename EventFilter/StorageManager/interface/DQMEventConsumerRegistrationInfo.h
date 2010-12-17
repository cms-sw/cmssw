// $Id: DQMEventConsumerRegistrationInfo.h,v 1.7 2010/12/16 16:35:29 mommsen Exp $
/// @file: DQMEventConsumerRegistrationInfo.h 

#ifndef StorageManager_DQMEventConsumerRegistrationInfo_h
#define StorageManager_DQMEventConsumerRegistrationInfo_h

#include <iosfwd>
#include <string>

#include "IOPool/Streamer/interface/HLTInfo.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/CommonRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor
{
  /**
   * Holds the registration information for a DQM event consumer.
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2010/12/16 16:35:29 $
   */

  class DQMEventConsumerRegistrationInfo : public RegistrationInfoBase
  {
  public:

    /**
     * Constructs an instance from the specified registration information.
     */
    DQMEventConsumerRegistrationInfo( const std::string& consumerName,
                                      const std::string& topLevelFolderName,
                                      const int& queueSize,
                                      const enquing_policy::PolicyTag& policy,
                                      const utils::duration_t& secondsToStale,
                                      const std::string& remoteHost );

    // Destructor:
    ~DQMEventConsumerRegistrationInfo();

    // Additional accessors:
    const std::string& topLevelFolderName() const { return _topLevelFolderName; }
    bool isProxyServer() const { return _isProxy; }
    const std::string& remoteHost() const { return _remoteHost; }

    // Staleness:
    bool isStale() const { return _stale; }
    void setStaleness( bool s ) { _stale = s; }

    // Comparison:
    bool operator<(const DQMEventConsumerRegistrationInfo&) const;
    bool operator==(const DQMEventConsumerRegistrationInfo&) const;
    bool operator!=(const DQMEventConsumerRegistrationInfo&) const;

    // Output:
    std::ostream& write(std::ostream& os) const;

    // Implementation of the Template Method pattern.
    virtual void do_registerMe(EventDistributor*);
    virtual QueueID do_queueId() const;
    virtual void do_setQueueId(QueueID const& id);
    virtual std::string do_consumerName() const;
    virtual ConsumerID do_consumerId() const;
    virtual void do_setConsumerId(ConsumerID const& id);
    virtual int do_queueSize() const;
    virtual enquing_policy::PolicyTag do_queuePolicy() const;
    virtual utils::duration_t do_secondsToStale() const;


  private:

    CommonRegistrationInfo _common;

    std::string _topLevelFolderName;
    bool _isProxy;
    bool _stale;

    std::string _remoteHost;

  };

  typedef boost::shared_ptr<stor::DQMEventConsumerRegistrationInfo> DQMEventConsRegPtr;


  /**
     Print the given DQMEventConsumerRegistrationInfo to the given
     stream.
  */
  inline
  std::ostream& operator<<(std::ostream& os, 
			   const DQMEventConsumerRegistrationInfo& ri)
  {
    return ri.write(os);
  }
  
} // namespace stor

#endif // StorageManager_DQMEventConsumerRegistrationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
