// $Id: DQMEventConsumerRegistrationInfo.h,v 1.2 2009/06/10 08:15:21 dshpakov Exp $
/// @file: DQMEventConsumerRegistrationInfo.h 

#ifndef StorageManager_DQMEventConsumerRegistrationInfo_h
#define StorageManager_DQMEventConsumerRegistrationInfo_h

#include <iosfwd>
#include <string>

#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/CommonRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor
{
  /**
   * Holds the registration information for a DQM event consumer.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:21 $
   */

  class DQMEventConsumerRegistrationInfo : public RegistrationInfoBase
  {
  public:

    /**
     * Constructs an instance from the specified registration information.
     */
    DQMEventConsumerRegistrationInfo( const std::string& consumerName,
                                      const std::string& topLevelFolderName,
                                      const size_t& queueSize,
                                      const enquing_policy::PolicyTag& policy,
                                      const utils::duration_t& secondsToStale );

    // Destructor:
    ~DQMEventConsumerRegistrationInfo();

    // Additional accessors:
    const std::string& topLevelFolderName() const { return _topLevelFolderName; }
    bool isProxyServer() const { return _isProxy; }

    // Staleness:
    bool isStale() const { return _stale; }
    void setStaleness( bool s ) { _stale = s; }

    // Output:
    std::ostream& write(std::ostream& os) const;

    // Implementation of the Template Method pattern.
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

    std::string _topLevelFolderName;
    bool _isProxy;
    bool _stale;
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
