// $Id: RegistrationCollection.h,v 1.8 2011/03/07 15:31:32 mommsen Exp $
/// @file: RegistrationCollection.h 

#ifndef EventFilter_StorageManager_RegistrationCollection_h
#define EventFilter_StorageManager_RegistrationCollection_h

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
     $Revision: 1.8 $
     $Date: 2011/03/07 15:31:32 $
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
    typedef std::vector<EventConsRegPtr> ConsumerRegistrations;
    void getEventConsumers( ConsumerRegistrations& ) const;

    /**
       Get DQM event consumer registrations.
    */
    typedef std::vector<DQMEventConsRegPtr> DQMConsumerRegistrations;
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
    bool registrationIsAllowed( const ConsumerID ) const;

  private:

    mutable boost::mutex lock_;

    ConsumerID nextConsumerId_;

    bool registrationAllowed_;
      
    typedef std::map<ConsumerID, RegPtr> RegistrationMap;
    RegistrationMap consumers_;

  };

  typedef boost::shared_ptr<RegistrationCollection> RegistrationCollectionPtr;

} // namespace stor

#endif // EventFilter_StorageManager_RegistrationCollection_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
