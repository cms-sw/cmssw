// $Id: RegistrationInfoBase.cc,v 1.4 2011/03/07 15:31:32 mommsen Exp $
/// @file: RegistrationInfoBase.cc

#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

using std::string;

namespace stor
{
  RegistrationInfoBase::RegistrationInfoBase
  (
    const std::string& consumerName,
    const std::string& remoteHost,
    const int& queueSize,
    const enquing_policy::PolicyTag& queuePolicy,
    const utils::Duration_t& secondsToStale
  ) :
  remoteHost_(remoteHost),
  consumerName_(consumerName),
  queueSize_(queueSize),
  queuePolicy_(queuePolicy),
  secondsToStale_(secondsToStale),
  consumerId_(0),
  lastConsumerContact_(utils::getCurrentTime())
  { }

  RegistrationInfoBase::RegistrationInfoBase
  (
    const edm::ParameterSet& pset,
    const std::string& remoteHost,
    const EventServingParams& eventServingParams,
    const bool useEventServingParams
  ) :
  remoteHost_(remoteHost),
  consumerId_(0),
  lastConsumerContact_(utils::getCurrentTime())
  {
    try
    {
      consumerName_ = pset.getUntrackedParameter<std::string>("consumerName");
    }
    catch( edm::Exception& e )
    {
      consumerName_ = pset.getUntrackedParameter<std::string>("DQMconsumerName", "Unknown");
    }

    try
    {
      sourceURL_ = pset.getParameter<std::string>("sourceURL");
    }
    catch( edm::Exception& e )
    {
      sourceURL_ = pset.getUntrackedParameter<std::string>("sourceURL", "Unknown");
    }

    const double maxEventRequestRate = pset.getUntrackedParameter<double>("maxEventRequestRate", 0);
    if ( maxEventRequestRate > 0 )
      minEventRequestInterval_ = utils::secondsToDuration(1 / maxEventRequestRate);
    else
      minEventRequestInterval_ = boost::posix_time::not_a_date_time;

    maxConnectTries_ = pset.getUntrackedParameter<int>("maxConnectTries", 300);

    connectTrySleepTime_ = pset.getUntrackedParameter<int>("connectTrySleepTime", 10);

    retryInterval_ = pset.getUntrackedParameter<int>("retryInterval", 5);

    queueSize_ = pset.getUntrackedParameter<int>("queueSize",
      useEventServingParams ? eventServingParams.consumerQueueSize_ : 0);

    const std::string policy =
      pset.getUntrackedParameter<std::string>("queuePolicy",
        useEventServingParams ? eventServingParams.consumerQueuePolicy_ : "Default");
    if ( policy == "DiscardNew" )
    {
      queuePolicy_ = enquing_policy::DiscardNew;
    }
    else if ( policy == "DiscardOld" )
    {
      queuePolicy_ = enquing_policy::DiscardOld;
    }
    else if ( policy == "Default" )
    {
      queuePolicy_ = enquing_policy::Max;
    }
    else
    {
      XCEPT_RAISE( stor::exception::ConsumerRegistration,
        "Unknown enqueuing policy: " + policy );
    }

    secondsToStale_ = utils::secondsToDuration(
      pset.getUntrackedParameter<double>("consumerTimeOut", 0)
    );
    if ( useEventServingParams && secondsToStale_ < boost::posix_time::seconds(1) )
      secondsToStale_ = eventServingParams.activeConsumerTimeout_;
  }

  edm::ParameterSet RegistrationInfoBase::getPSet() const
  {
    edm::ParameterSet pset;

    if ( consumerName_ != "Unknown" )
      pset.addUntrackedParameter<std::string>("consumerName", consumerName_);

    if ( sourceURL_  != "Unknown" )
      pset.addParameter<std::string>("sourceURL", sourceURL_);

    if ( maxConnectTries_ != 300 )
      pset.addUntrackedParameter<int>("maxConnectTries", maxConnectTries_);
    
    if ( connectTrySleepTime_ != 10 )
      pset.addUntrackedParameter<int>("connectTrySleepTime", connectTrySleepTime_);

    if ( retryInterval_ != 5 )
      pset.addUntrackedParameter<int>("retryInterval", retryInterval_);
    
    if ( queueSize_ > 0 )
      pset.addUntrackedParameter<int>("queueSize", queueSize_);
    
    if ( ! minEventRequestInterval_.is_not_a_date_time() )
    {
      const double rate = 1 / utils::durationToSeconds(minEventRequestInterval_);
      pset.addUntrackedParameter<double>("maxEventRequestRate", rate);
    }

    const double secondsToStale = utils::durationToSeconds(secondsToStale_);
    if ( secondsToStale > 0 )
      pset.addUntrackedParameter<double>("consumerTimeOut", secondsToStale);

    if ( queuePolicy_ == enquing_policy::DiscardNew )
      pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
    if ( queuePolicy_ == enquing_policy::DiscardOld )
      pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");

    do_appendToPSet(pset);

    return pset;
  }

  bool RegistrationInfoBase::operator<(const RegistrationInfoBase& other) const
  {
    if ( queueSize() != other.queueSize() )
      return ( queueSize() < other.queueSize() );
    if ( queuePolicy() != other.queuePolicy() )
      return ( queuePolicy() < other.queuePolicy() );
    return ( secondsToStale() < other.secondsToStale() );
  }

  bool RegistrationInfoBase::operator==(const RegistrationInfoBase& other) const
  {
    return (
      queueSize() == other.queueSize() &&
      queuePolicy() == other.queuePolicy() &&
      secondsToStale() == other.secondsToStale()
    );
  }

  bool RegistrationInfoBase::operator!=(const RegistrationInfoBase& other) const
  {
    return ! ( *this == other );
  }

  void RegistrationInfoBase::queueInfo(std::ostream& os) const
  {
    os << "Queue type: " << queuePolicy_ <<
      ", size " << queueSize_ << 
      ", timeout " << secondsToStale_.total_seconds() << "s";
  }

  std::ostream& operator<< (std::ostream& os,
                            RegistrationInfoBase const& ri)
  {
    os << "\n Consumer name: " << ri.consumerName()
      << "\n Consumer id: " << ri.consumerId()
      << "\n Source URL: " << ri.sourceURL()
      << "\n Remote Host: " << ri.remoteHost()
      << "\n Queue id: " << ri.queueId()
      << "\n Maximum size of queue: " << ri.queueSize()
      << "\n Policy used if queue is full: " << ri.queuePolicy()
      << "\n Time until queue becomes stale (seconds): " << ri.secondsToStale().total_seconds();
    return os;
  }
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
