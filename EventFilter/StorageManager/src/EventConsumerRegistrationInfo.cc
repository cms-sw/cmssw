// $Id: EventConsumerRegistrationInfo.cc,v 1.15 2011/03/07 15:31:32 mommsen Exp $
/// @file: EventConsumerRegistrationInfo.cc

#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <iterator>
#include <ostream>


namespace stor
{
  EventConsumerRegistrationInfo::EventConsumerRegistrationInfo
  (
    const edm::ParameterSet& pset,
    const EventServingParams& eventServingParams,
    const std::string& remoteHost
  ) :
  RegistrationInfoBase(pset, remoteHost, eventServingParams, true)
  {
    parsePSet(pset);
  }

  EventConsumerRegistrationInfo::EventConsumerRegistrationInfo
  (
    const edm::ParameterSet& pset,
    const std::string& remoteHost
  ) :
  RegistrationInfoBase(pset, remoteHost, EventServingParams(), false)
  {
    parsePSet(pset);
  }

  void 
  EventConsumerRegistrationInfo::parsePSet(const edm::ParameterSet& pset)
  {
    try
    {
      outputModuleLabel_ = pset.getUntrackedParameter<std::string>("SelectHLTOutput");
    }
    catch( edm::Exception& e )
    {
      XCEPT_RAISE( stor::exception::ConsumerRegistration,
        "No HLT output module specified" );
    }

    triggerSelection_ = pset.getUntrackedParameter<std::string>("TriggerSelector", "");

    try
    {
      eventSelection_ = pset.getParameter<Strings>("TrackedEventSelection");
    }
    catch( edm::Exception& e )
    {
      edm::ParameterSet tmpPSet1 =
        pset.getUntrackedParameter<edm::ParameterSet>("SelectEvents", edm::ParameterSet());
      if ( ! tmpPSet1.empty() )
      {
        eventSelection_ = tmpPSet1.getParameter<Strings>("SelectEvents");
      }
    }

    prescale_ = pset.getUntrackedParameter<int>("prescale", 1);
    uniqueEvents_ = pset.getUntrackedParameter<bool>("uniqueEvents", false);
    headerRetryInterval_ = pset.getUntrackedParameter<int>("headerRetryInterval", 5);
  }
  
  
  void
  EventConsumerRegistrationInfo::do_appendToPSet(edm::ParameterSet& pset) const
  {
    pset.addUntrackedParameter<std::string>("SelectHLTOutput", outputModuleLabel_);
    pset.addUntrackedParameter<std::string>("TriggerSelector", triggerSelection_);
    pset.addParameter<Strings>("TrackedEventSelection", eventSelection_);
    pset.addUntrackedParameter<bool>("uniqueEvents", uniqueEvents_);
    pset.addUntrackedParameter<int>("prescale", prescale_);

    if ( headerRetryInterval_ != 5 )
      pset.addUntrackedParameter<int>("headerRetryInterval", headerRetryInterval_);
  }

  void 
  EventConsumerRegistrationInfo::do_registerMe(EventDistributor* evtDist)
  {
    evtDist->registerEventConsumer(shared_from_this());
  }
  
  void
  EventConsumerRegistrationInfo::do_eventType(std::ostream& os) const
  {
    os << "Output module: " << outputModuleLabel_ << "\n";

    if ( triggerSelection_.empty() )
    {
      if ( ! eventSelection_.empty() )
      {
        os  << "Event Selection: ";
        std::copy(eventSelection_.begin(), eventSelection_.end()-1,
          std::ostream_iterator<Strings::value_type>(os, ","));
        os << *(eventSelection_.end()-1);
      }
    }
    else
      os << "Trigger Selection: " << triggerSelection_;

    if ( prescale_ != 1 )
      os << "; prescale: " << prescale_;

    if ( uniqueEvents_ != false )
      os << "; uniqueEvents";

    os << "\n";
    queueInfo(os);
  }
  
  bool
  EventConsumerRegistrationInfo::operator<(const EventConsumerRegistrationInfo& other) const
  {
    if ( outputModuleLabel() != other.outputModuleLabel() )
      return ( outputModuleLabel() < other.outputModuleLabel() );
    if ( triggerSelection() != other.triggerSelection() )
      return ( triggerSelection() < other.triggerSelection() );
    if ( eventSelection() != other.eventSelection() )
      return ( eventSelection() < other.eventSelection() );
    if ( prescale() != other.prescale() )
      return ( prescale() < other.prescale() );
    if ( uniqueEvents() != other.uniqueEvents() )
      return ( uniqueEvents() < other.uniqueEvents() );
    return RegistrationInfoBase::operator<(other);
  }

  bool
  EventConsumerRegistrationInfo::operator==(const EventConsumerRegistrationInfo& other) const
  {
    return (
      outputModuleLabel() == other.outputModuleLabel() &&
      triggerSelection() == other.triggerSelection() &&
      eventSelection() == other.eventSelection() &&
      prescale() == other.prescale() &&
      uniqueEvents() == other.uniqueEvents() &&
      RegistrationInfoBase::operator==(other)
    );
  }

  bool
  EventConsumerRegistrationInfo::operator!=(const EventConsumerRegistrationInfo& other) const
  {
    return ! ( *this == other );
  }

}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
