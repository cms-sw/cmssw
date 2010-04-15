// $Id: RegistrationCollection.h,v 1.3 2009/07/20 13:06:10 mommsen Exp $
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

     $Author: mommsen $
     $Revision: 1.3 $
     $Date: 2009/07/20 13:06:10 $
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
