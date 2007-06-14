#ifndef FWCore_Framework_EventHelperDescription_h
#define FWCore_Framework_EventHelperDescription_h
// -*- C++ -*-
//
// Package:     <package>
// Module:      EventHelperDescription
// 
/**\class EventHelperDescription EventHelperDescription.h package/EventHelperDescription.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Valentin Kuznetsov
// Created:     Thu Jul 13 10:40:21 EDT 2006
// $Id: EventHelperDescription.h,v 1.1 2006/07/23 01:24:33 valya Exp $
//
// Revision history
//
// $Log: EventHelperDescription.h,v $
// Revision 1.1  2006/07/23 01:24:33  valya
// Add looper support into framework. The base class is EDLooper. All the work done in EventProcessor and EventHelperLooper
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventPrincipal.h"

// forward declarations

namespace edm {

struct EventHelperDescription
{
      EventHelperDescription()
           : eventPrincipal_(), eventSetup_(0) {}
           
      EventHelperDescription(std::auto_ptr<edm::EventPrincipal> p, const edm::EventSetup* s)
           : eventPrincipal_(p), eventSetup_(s) {}

      mutable std::auto_ptr<edm::EventPrincipal> eventPrincipal_;
      const edm::EventSetup*     eventSetup_;

      EventHelperDescription(const EventHelperDescription& iOther) :
           eventPrincipal_(iOther.eventPrincipal_),
           eventSetup_(iOther.eventSetup_) {}

      EventHelperDescription& operator=(EventHelperDescription& iOther) {
           eventPrincipal_ = iOther.eventPrincipal_;
           eventSetup_ = iOther.eventSetup_;
           return (*this);
      }
};

// inline function definitions

} // end of namespace

#endif /* FWCORE_FRAMEWORK_EVENTHELPERDESCRIPTION_H */
