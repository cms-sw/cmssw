// $Id: EventConsumerRegistrationInfo.h,v 1.15 2011/08/31 20:11:59 wmtan Exp $
/// @file: EventConsumerRegistrationInfo.h 

#ifndef EventFilter_StorageManager_EventConsumerRegistrationInfo_h
#define EventFilter_StorageManager_EventConsumerRegistrationInfo_h

#include <iosfwd>
#include <string>

#include <boost/shared_ptr.hpp>

#include "toolbox/net/Utils.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include <boost/enable_shared_from_this.hpp>


namespace stor
{
  /**
   * Holds the registration information from a event consumer.
   *
   * $Author: wmtan $
   * $Revision: 1.15 $
   * $Date: 2011/08/31 20:11:59 $
   */

  class EventConsumerRegistrationInfo :
    public RegistrationInfoBase,
    public boost::enable_shared_from_this<EventConsumerRegistrationInfo>
  {

  public:
    
    EventConsumerRegistrationInfo
    (
      const edm::ParameterSet& pset,
      const EventServingParams& eventServingParams,
      const std::string& remoteHost = toolbox::net::getHostName()
    );

    EventConsumerRegistrationInfo
    (
      const edm::ParameterSet& pset,
      const std::string& remoteHost = toolbox::net::getHostName()
    );

    ~EventConsumerRegistrationInfo() {};

    // Accessors:
    const std::string& triggerSelection() const { return triggerSelection_; }
    const Strings& eventSelection() const { return eventSelection_; }
    const std::string& outputModuleLabel() const { return outputModuleLabel_; }
    const int& prescale() const { return prescale_; }
    const bool& uniqueEvents() const { return uniqueEvents_; }
    const int& headerRetryInterval() const { return headerRetryInterval_; }
    uint32 eventRequestCode() const { return Header::EVENT_REQUEST; }
    uint32 eventCode() const { return Header::EVENT; }
    std::string eventURL() const { return sourceURL() + "/geteventdata"; }
    std::string registerURL() const { return sourceURL() + "/registerConsumer"; }

    // Comparison:
    bool operator<(const EventConsumerRegistrationInfo&) const;
    bool operator==(const EventConsumerRegistrationInfo&) const;
    bool operator!=(const EventConsumerRegistrationInfo&) const;

    // Output:
    std::ostream& write(std::ostream& os) const;

    // Implementation of Template Method pattern.
    virtual void do_registerMe(EventDistributor*);
    virtual void do_eventType(std::ostream&) const;
    virtual void do_appendToPSet(edm::ParameterSet&) const;

  private:

    void parsePSet(const edm::ParameterSet&);

    std::string triggerSelection_;
    Strings eventSelection_;
    std::string outputModuleLabel_;
    int prescale_;
    bool uniqueEvents_;
    int headerRetryInterval_;
  };

  typedef boost::shared_ptr<stor::EventConsumerRegistrationInfo> EventConsRegPtr;

} // namespace stor

#endif // EventFilter_StorageManager_EventConsumerRegistrationInfo_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
