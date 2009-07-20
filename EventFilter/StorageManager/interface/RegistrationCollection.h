// $Id: RegistrationCollection.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $
/// @file: RegistrationCollection.h 

#ifndef StorageManager_RegistrationCollection_h
#define StorageManager_RegistrationCollection_h

#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"

#include <vector>
#include <map>

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

namespace stor
{

  /**
     Keep a collection of registered event and DQM event consumers.

     $Author: dshpakov $
     $Revision: 1.4 $
     $Date: 2009/07/14 10:34:44 $
  */

  class RegistrationCollection
  {

  public:

    RegistrationCollection();

    ~RegistrationCollection();

    /**
       Return next available consumer ID or 0 if no registration is
       allowed.
    */
    ConsumerID getConsumerID();

    /**
       Add registration info. Return false if no registration is allowed.
    */
    bool addRegistrationInfo( ConsumerID, RegPtr );

    /**
       Get event consumer registrations.
    */
    typedef std::vector<stor::ConsRegPtr> ConsumerRegistrations;
    void getEventConsumers( ConsumerRegistrations& );

    /**
       Get DQM event consumer registrations.
    */
    typedef std::vector<stor::DQMEventConsRegPtr> DQMConsumerRegistrations;
    void getDQMEventConsumers( DQMConsumerRegistrations& );

    /**
       Enable registration.
    */
    void enableConsumerRegistration();

    /**
       Disable registration.
    */
    void disableConsumerRegistration();

    /**
       Clear registrations.
    */
    void clearRegistrations();

    /**
       Test if registration is allowed.
    */
    bool registrationIsAllowed() const;

    /**
       Test if consumer is a proxy.
    */
    bool isProxy( ConsumerID ) const;

  private:

    mutable boost::mutex _lock;

    ConsumerID _nextConsumerID;

    bool _registrationAllowed;
      
    typedef std::map<ConsumerID, RegPtr> RegistrationMap;
    RegistrationMap _consumers;

  };

} // namespace stor

#endif // StorageManager_RegistrationCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
