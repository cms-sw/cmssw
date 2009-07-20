// $Id: RegistrationInfoBase.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $
/// @file: RegistrationInfoBase.h 

#ifndef StorageManager_RegistrationInfoBase_h
#define StorageManager_RegistrationInfoBase_h

#include <string>

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor {

  class EventDistributor;

  /**
   * Defines the common interface for event and DQM consumer
   * registration info objects.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */

  class RegistrationInfoBase
  {

  public:

    /**
       The virtual destructor allows polymorphic
       containment-by-reference.
    */
    virtual ~RegistrationInfoBase() {};

    /**
       Register the consumer represented by this registration with the
       specified EventDistributor.
    */
    void registerMe(EventDistributor* dist);

    /**
     * Returns the ID of the queue corresponding to this consumer
     * registration.
     */
    QueueID queueId() const;
    
    /**
       Set the consumer ID.
     */
    void setQueueID(QueueID const& id);

    /**
     * Returns the enquing policy requested by the consumer
     * registration.
     */
    enquing_policy::PolicyTag queuePolicy() const;

    /**
       Returns the name supplied by the consumer.
     */
    std::string consumerName() const;

    /**
       Returns the ID given to this consumer.
     */
    ConsumerID consumerId() const;

    /**
       Set the consumer ID.
     */
    void setConsumerID(const ConsumerID& id);

    /**
       Returns the queue Size
     */
    size_t queueSize() const;

    /**
       Returns the time until the queue becomes stale
     */
    utils::duration_t secondsToStale() const;


  private:
    virtual void do_registerMe(EventDistributor*) = 0;
    virtual QueueID do_queueId() const = 0;
    virtual void do_setQueueID(QueueID const& id) = 0;
    virtual std::string do_consumerName() const = 0;
    virtual ConsumerID do_consumerId() const = 0;
    virtual void do_setConsumerID(ConsumerID const& id) = 0;
    virtual size_t do_queueSize() const = 0;
    virtual enquing_policy::PolicyTag do_queuePolicy() const = 0;
    virtual utils::duration_t do_secondsToStale() const = 0;
  };

  typedef boost::shared_ptr<stor::RegistrationInfoBase> RegPtr;

  inline
  void RegistrationInfoBase::registerMe(EventDistributor* dist)
  {
    do_registerMe(dist);
  }

  inline
  QueueID RegistrationInfoBase::queueId() const
  {
    return do_queueId();
  }

  inline
  void RegistrationInfoBase::setQueueID(QueueID const& id)
  {
    do_setQueueID(id);
  }

  inline
  enquing_policy::PolicyTag RegistrationInfoBase::queuePolicy() const
  {
    return do_queuePolicy();
  }

  inline
  std::string RegistrationInfoBase::consumerName() const
  {
    return do_consumerName();
  }

  inline
  ConsumerID RegistrationInfoBase::consumerId() const
  {
    return do_consumerId();
  }

  inline
  void RegistrationInfoBase::setConsumerID(ConsumerID const& id)
  {
    do_setConsumerID(id);
  }

  inline
  size_t RegistrationInfoBase::queueSize() const
  {
    return do_queueSize();
  }

  inline
  utils::duration_t RegistrationInfoBase::secondsToStale() const
  {
    return do_secondsToStale();
  }

} // namespace stor

#endif // StorageManager_RegistrationInfoBase_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
