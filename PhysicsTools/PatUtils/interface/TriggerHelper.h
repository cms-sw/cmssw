#ifndef PhysicsTools_PatUtils_TriggerHelper_h
#define PhysicsTools_PatUtils_TriggerHelper_h


// -*- C++ -*-
//
// Package:    PatUtils
// Class:      pat::helper::TriggerHelper
//
// $Id: TriggerHelper.h,v 1.7 2011/04/05 19:41:33 vadler Exp $
//
/**
  \class    pat::helper::TriggerHelper TriggerHelper.h "PhysicsTools/PatUtils/interface/TriggerHelper.h"
  \brief    Helper class to remove unwanted dependencies from DataFormats/PatCandidates related to PAT trigger data formats

            The following class(es) are implemented in the pat::helper namespace:
            - TriggerMatchHelper:
              + provides the usage of functions which need the edm::AssociativeIterator;

  \author   Volker Adler
  \version  $Id: TriggerHelper.h,v 1.7 2011/04/05 19:41:33 vadler Exp $
*/


#include <string>

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"


namespace pat {

  namespace helper {

    class TriggerMatchHelper {

      public:

        /// Constructors and Destructor

        /// Default constructor
        TriggerMatchHelper() {};

        /// Destructor
        ~TriggerMatchHelper() {};

        /// Methods

        /// Get a reference to the trigger objects matched to a certain physics object given by a reference for a certain matcher module
        /// ... by resulting association
        TriggerObjectRef triggerMatchObject( const reco::CandidateBaseRef & candRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// ... by matcher module label
        TriggerObjectRef triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher      , const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// Get a reference to the trigger objects matched to a certain physics object given by a collection and index for a certain matcher module
        /// ... by resulting association
        template< class C > TriggerObjectRef triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// ... by matcher module label
        template< class C > TriggerObjectRef triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const std::string & labelMatcher      , const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// Get a table of references to all trigger objects matched to a certain physics object given by a reference
        TriggerObjectMatchMap triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// Get a table of references to all trigger objects matched to a certain physics object given by a collection and index
        template< class C > TriggerObjectMatchMap triggerMatchObjects( const edm::Handle< C > & candCollHandle, const size_t iCand, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// Get a vector of references to the phyics objects matched to a certain trigger object given by a reference for a certain matcher module
        /// ... by resulting association
        reco::CandidateBaseRefVector triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// ... by matcher module label
        reco::CandidateBaseRefVector triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const std::string & labelMatcher      , const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// Get a vector of references to the phyics objects matched to a certain trigger object given by a collection and index for a certain matcher module
        /// ... by resulting association
        reco::CandidateBaseRefVector triggerMatchCandidates( const edm::Handle< TriggerObjectCollection > & trigCollHandle, const size_t iTrig, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        /// ... by matcher module label
        reco::CandidateBaseRefVector triggerMatchCandidates( const edm::Handle< TriggerObjectCollection > & trigCollHandle, const size_t iTrig, const std::string & labelMatcher      , const edm::Event & event, const TriggerEvent & triggerEvent ) const;

    };

    // Method Templates

    // Get a reference to the trigger objects matched to a certain physics object given by a collection and index for a certain matcher module
    template< class C > TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      const reco::CandidateBaseRef candRef( edm::Ref< C >( candCollHandle, iCand ) );
      return triggerMatchObject( candRef, matchResult, event, triggerEvent );
    }
    template< class C > TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      return triggerMatchObject( candCollHandle, iCand, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
    }

    // Get a table of references to all trigger objects matched to a certain physics object given by a collection and index
    template< class C > TriggerObjectMatchMap TriggerMatchHelper::triggerMatchObjects( const edm::Handle< C > & candCollHandle, const size_t iCand, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      const reco::CandidateBaseRef candRef( edm::Ref< C >( candCollHandle, iCand ) );
      return triggerMatchObjects( candRef, event, triggerEvent );
    }

  }

}


#endif
