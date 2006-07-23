#if !defined(FWCORE_FRAMEWORK_EVENTHELPERDESCRIPTION_H)
#define FWCORE_FRAMEWORK_EVENTHELPERDESCRIPTION_H
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
// $Id$
//
// Revision history
//
// $Log$

// system include files

// user include files
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"

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
