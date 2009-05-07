#ifndef PhysicsTools_PatUtils_TriggerHelper_h
#define PhysicsTools_PatUtils_TriggerHelper_h


// -*- C++ -*-
//
// Package:    PatUtils
// Class:      pat::helper::TriggerHelper
//
// $Id$
//
/**
  \class    pat::helper::TriggerHelper TriggerHelper.h "PhysicsTools/PatUtils/interface/TriggerHelper.h"
  \brief    Helper class to remove unwanted dependencies from DataFormats/PatCandidates related to PAT trigger data formats

            The following class(es) are implemented in the pat::helper namespace:
            - TriggerMatchHelper:
              + provides the usage of functions which need the edm::AssociativeIterator;

  \author   Volker Adler
  \version  $Id$
*/


#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"

#include <string>

#include "DataFormats/PatCandidates/interface/TriggerObject.h"
#include "DataFormats/Candidate/interface/Candidate.h"


namespace pat {

  namespace helper {

    class TriggerMatchHelper {

      public:

        /// constructors and destructor
        TriggerMatchHelper() {};
        ~TriggerMatchHelper() {};

        /// functions
        pat::TriggerObjectRef        triggerMatchObject( const reco::CandidateBaseRef & candRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        pat::TriggerObjectRef        triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        pat::TriggerObjectMatchMap   triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        reco::CandidateBaseRefVector triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        reco::CandidateBaseRefVector triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const;

    };

  }

}


#endif
