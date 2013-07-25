// $Id: DQMEventConsumerRegistrationInfo.h,v 1.9 2011/03/07 15:31:31 mommsen Exp $
/// @file: DQMEventConsumerRegistrationInfo.h 

#ifndef EventFilter_StorageManager_DQMEventConsumerRegistrationInfo_h
#define EventFilter_StorageManager_DQMEventConsumerRegistrationInfo_h

#include <iosfwd>
#include <string>

#include <boost/shared_ptr.hpp>

#include "toolbox/net/Utils.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include <boost/enable_shared_from_this.hpp>


namespace stor
{
  /**
   * Holds the registration information for a DQM event consumer.
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2011/03/07 15:31:31 $
   */

  class DQMEventConsumerRegistrationInfo : 
    public RegistrationInfoBase,
    public boost::enable_shared_from_this<DQMEventConsumerRegistrationInfo>
  {
  public:

    /**
     * Constructs an instance from the specified registration information.
     */
    DQMEventConsumerRegistrationInfo
    (
      const edm::ParameterSet& pset,
      const EventServingParams& eventServingParams,
      const std::string& remoteHost = toolbox::net::getHostName()
    );

    DQMEventConsumerRegistrationInfo
    (
      const edm::ParameterSet& pset,
      const std::string& remoteHost = toolbox::net::getHostName()
    );

    // Destructor:
    ~DQMEventConsumerRegistrationInfo() {};

    // Accessors:
    const std::string& topLevelFolderName() const { return topLevelFolderName_; }
    uint32 eventRequestCode() const { return Header::DQMEVENT_REQUEST; }
    uint32 eventCode() const { return Header::DQM_EVENT; }
    std::string eventURL() const { return sourceURL() + "/getDQMeventdata"; }
    std::string registerURL() const { return sourceURL() + "/registerDQMConsumer"; }

    // Comparison:
    bool operator<(const DQMEventConsumerRegistrationInfo&) const;
    bool operator==(const DQMEventConsumerRegistrationInfo&) const;
    bool operator!=(const DQMEventConsumerRegistrationInfo&) const;

    // Output:
    std::ostream& write(std::ostream& os) const;

    // Implementation of the Template Method pattern.
    virtual void do_registerMe(EventDistributor*);
    virtual void do_eventType(std::ostream&) const;
    virtual void do_appendToPSet(edm::ParameterSet&) const;

  private:

    void parsePSet(const edm::ParameterSet&);

    std::string topLevelFolderName_;
  };

  typedef boost::shared_ptr<stor::DQMEventConsumerRegistrationInfo> DQMEventConsRegPtr;
  
} // namespace stor

#endif // EventFilter_StorageManager_DQMEventConsumerRegistrationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
