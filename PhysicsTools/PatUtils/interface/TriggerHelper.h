#ifndef PhysicsTools_PatUtils_TriggerHelper_h
#define PhysicsTools_PatUtils_TriggerHelper_h


// -*- C++ -*-
//
// Package:    PatUtils
// Class:      pat::helper::TriggerHelper
//
// $Id: TriggerHelper.h,v 1.4 2010/07/01 21:27:49 vadler Exp $
//
/**
  \class    pat::helper::TriggerHelper TriggerHelper.h "PhysicsTools/PatUtils/interface/TriggerHelper.h"
  \brief    Helper class to remove unwanted dependencies from DataFormats/PatCandidates related to PAT trigger data formats

            The following class(es) are implemented in the pat::helper namespace:
            - TriggerMatchHelper:
              + provides the usage of functions which need the edm::AssociativeIterator;

  \author   Volker Adler
  \version  $Id: TriggerHelper.h,v 1.4 2010/07/01 21:27:49 vadler Exp $
*/


#include <string>

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"


namespace pat {

  namespace helper {

    class TriggerMatchHelper {

      public:

        /// constructors and destructor
        TriggerMatchHelper() {};
        ~TriggerMatchHelper() {};

        /// functions
        TriggerObjectRef                          triggerMatchObject( const reco::CandidateBaseRef & candRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        TriggerObjectRef                          triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        template< class C > TriggerObjectRef      triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        template< class C > TriggerObjectRef      triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        TriggerObjectMatchMap                     triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        template< class C > TriggerObjectMatchMap triggerMatchObjects( const edm::Handle< C > & candCollHandle, const size_t iCand, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        reco::CandidateBaseRefVector              triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const;
        reco::CandidateBaseRefVector              triggerMatchCandidates( const pat::TriggerObjectRef & objectRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const;

    };

    /// function templates
    template< class C > TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      const reco::CandidateBaseRef candRef( edm::Ref< C >( candCollHandle, iCand ) );
      return triggerMatchObject( candRef, matchResult, event, triggerEvent );
    }
    template< class C > TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const edm::Handle< C > & candCollHandle, const size_t iCand, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      return triggerMatchObject( candCollHandle, iCand, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
    }
    template< class C > TriggerObjectMatchMap TriggerMatchHelper::triggerMatchObjects( const edm::Handle< C > & candCollHandle, const size_t iCand, const edm::Event & event, const TriggerEvent & triggerEvent ) const
    {
      const reco::CandidateBaseRef candRef( edm::Ref< C >( candCollHandle, iCand ) );
      return triggerMatchObjects( candRef, event, triggerEvent );
    }

  }

}


#endif
