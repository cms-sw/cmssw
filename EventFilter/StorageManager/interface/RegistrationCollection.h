// $Id: RegistrationCollection.h,v 1.6 2010/04/16 14:39:34 mommsen Exp $
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
     $Revision: 1.6 $
     $Date: 2010/04/16 14:39:34 $
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
    ConsumerID getConsumerId();

    /**
       Add registration info. Return false if no registration is allowed.
    */
    bool addRegistrationInfo( RegPtr );

    /**
       Get registration info for ConsumerID. Returns empty pointer if not found.
    */
    RegPtr getRegistrationInfo( const ConsumerID ) const;

    /**
       Get event consumer registrations.
    */
    typedef std::vector<stor::EventConsRegPtr> ConsumerRegistrations;
    void getEventConsumers( ConsumerRegistrations& ) const;

    /**
       Get DQM event consumer registrations.
    */
    typedef std::vector<stor::DQMEventConsRegPtr> DQMConsumerRegistrations;
    void getDQMEventConsumers( DQMConsumerRegistrations& ) const;

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

    ConsumerID _nextConsumerId;

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
