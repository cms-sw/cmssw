// $Id: RegistrationInfoBase.h,v 1.7 2011/03/07 15:31:32 mommsen Exp $
/// @file: RegistrationInfoBase.h 

#ifndef EventFilter_StorageManager_RegistrationInfoBase_h
#define EventFilter_StorageManager_RegistrationInfoBase_h

#include <string>

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace stor {

  class EventDistributor;

  /**
   * Defines the common interface for event and DQM consumer
   * registration info objects.
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2011/03/07 15:31:32 $
   */

  class RegistrationInfoBase
  {

  public:
    
    RegistrationInfoBase
    (
      const std::string& consumerName,
      const std::string& remoteHost,
      const int& queueSize,
      const enquing_policy::PolicyTag& queuePolicy,
      const utils::Duration_t& secondsToStale
    );

    RegistrationInfoBase
    (
      const edm::ParameterSet& pset,
      const std::string& remoteHost,
      const EventServingParams& eventServingParams,
      const bool useEventServingParams
    );

    /**
       The virtual destructor allows polymorphic
       containment-by-reference.
    */
    virtual ~RegistrationInfoBase() {};

    /**
       Mark time when consumer last contacted us
    */
    void consumerContact();

    /**
       Register the consumer represented by this registration with the
       specified EventDistributor.
    */
    void registerMe(EventDistributor* dist);

    /**
       Returns a formatted string which contains the information about the event type.
     */
    void eventType(std::ostream&) const;

    /**
      Return the ParameterSet containing the consumer registration infos
    */
    edm::ParameterSet getPSet() const;

    /**
      Print queue information into ostream
     */
    void queueInfo(std::ostream&) const;

    // Setters:
    void setMinEventRequestInterval(const utils::Duration_t& interval) { minEventRequestInterval_ = interval; }

    // Accessors
    bool isValid() const { return consumerId_.isValid(); }
    const QueueID& queueId() const { return queueId_; }
    const enquing_policy::PolicyTag& queuePolicy() const { return queuePolicy_; }
    const std::string& consumerName() const { return consumerName_; }
    const std::string& remoteHost() const { return remoteHost_; }
    const std::string& sourceURL() const { return sourceURL_; }
    const ConsumerID& consumerId() const { return consumerId_; }
    const int& queueSize() const { return queueSize_; }
    const int& maxConnectTries() const { return maxConnectTries_; }
    const int& connectTrySleepTime() const { return connectTrySleepTime_; }
    const int& retryInterval() const { return retryInterval_; }
    const utils::Duration_t& minEventRequestInterval() const { return minEventRequestInterval_; }
    const utils::Duration_t& secondsToStale() const { return secondsToStale_; }
    bool isStale(const utils::TimePoint_t&) const;
    double lastContactSecondsAgo(const utils::TimePoint_t&) const;

    // Setters
    void setQueueId(const QueueID& id) { queueId_ = id; }
    void setSourceURL(const std::string& url) { sourceURL_ = url; }
    void setConsumerId(const ConsumerID& id) { consumerId_ = id; }

    // Comparison:
    virtual bool operator<(const RegistrationInfoBase&) const;
    virtual bool operator==(const RegistrationInfoBase&) const;
    virtual bool operator!=(const RegistrationInfoBase&) const;


  protected:

    virtual void do_registerMe(EventDistributor*) = 0;
    virtual void do_eventType(std::ostream&) const = 0;
    virtual void do_appendToPSet(edm::ParameterSet&) const = 0;


  private:

    const std::string                remoteHost_;
    std::string                      consumerName_;
    std::string                      sourceURL_;
    int                              queueSize_;
    enquing_policy::PolicyTag        queuePolicy_;
    utils::Duration_t                secondsToStale_;
    int                              maxConnectTries_;
    int                              connectTrySleepTime_;
    int                              retryInterval_;
    utils::Duration_t                minEventRequestInterval_;
    QueueID                          queueId_;
    ConsumerID                       consumerId_;
    utils::TimePoint_t               lastConsumerContact_;
  };

  typedef boost::shared_ptr<RegistrationInfoBase> RegPtr;


  inline
  void RegistrationInfoBase::consumerContact()
  {
    lastConsumerContact_ = utils::getCurrentTime();
  }

  inline
  void RegistrationInfoBase::registerMe(EventDistributor* dist)
  {
    do_registerMe(dist);
  }

  inline
  void RegistrationInfoBase::eventType(std::ostream& os) const
  {
    do_eventType(os);
  }

  inline
  bool RegistrationInfoBase::isStale(const utils::TimePoint_t& now) const
  {
    return ( now > lastConsumerContact_ + secondsToStale() );
  }

  inline
  double RegistrationInfoBase::lastContactSecondsAgo(const utils::TimePoint_t& now) const
  {
    return utils::durationToSeconds( now - lastConsumerContact_ );
  }

  std::ostream& operator<<(std::ostream& os, 
                           RegistrationInfoBase const& ri);

} // namespace stor

#endif // EventFilter_StorageManager_RegistrationInfoBase_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
