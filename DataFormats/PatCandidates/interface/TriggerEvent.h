#ifndef DataFormats_PatCandidates_TriggerEvent_h
#define DataFormats_PatCandidates_TriggerEvent_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerEvent
//
// $Id: TriggerEvent.h,v 1.1.2.19 2009/03/18 19:04:52 vadler Exp $
//
/**
  \class    pat::TriggerEvent TriggerEvent.h "DataFormats/PatCandidates/interface/TriggerEvent.h"
  \brief    Analysis-level trigger object class

   TriggerEvent implements a container for trigger event's information within the 'pat' namespace.
   It has the following data members:
   - [to be filled]

  \author   Volker Adler
  \version  $Id: TriggerEvent.h,v 1.1.2.19 2009/03/18 19:04:52 vadler Exp $
*/


#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
   

namespace pat {

  class TriggerEvent {
    
      /// event related data members
      std::string nameHltTable_;
      bool        run_;
      bool        accept_;
      bool        error_;
      
      /// paths related data members
      TriggerPathRefProd paths_;
      
      /// filters related data members
      TriggerFilterRefProd filters_;
      
      /// objects related data members
      TriggerObjectRefProd        objects_;
      TriggerObjectMatchContainer objectMatchResults_;

    public:
    
      /// constructors and desctructor
      TriggerEvent() { objectMatchResults_.clear(); };
      TriggerEvent( const std::string & nameHltTable, bool run, bool accept, bool error );
      virtual ~TriggerEvent() {};
      
      /// event related
      void setNameHltTable( const std::string & name ) { nameHltTable_ = name; };
      void setRun( bool run )                          { run_          = run; };
      void setAccept( bool accept )                    { accept_       = accept; };
      void setError( bool error )                      { error         = error; };
      std::string nameHltTable() const { return nameHltTable_; };
      bool        wasRun() const       { return run_; };
      bool        wasAccept() const    { return accept_; };
      bool        wasError() const     { return error_; };
      
      /// paths related
      void setPaths( const edm::Handle< TriggerPathCollection > & handleTriggerPaths ) { paths_ = TriggerPathRefProd( handleTriggerPaths ); };
      const TriggerPathCollection * paths() const { return paths_.get(); };          // returns 0 if RefProd is null
      const TriggerPath           * path( const std::string & namePath ) const;      // returns 0 if path is not found
      unsigned                      indexPath( const std::string & namePath ) const; // returns size of path collection if path is not found
      TriggerPathRefVector          acceptedPaths() const;                           // transient
      
      /// filters related
      void setFilters( const edm::Handle< TriggerFilterCollection > & handleTriggerFilters ) { filters_ = TriggerFilterRefProd( handleTriggerFilters ); };
      const TriggerFilterCollection * filters() const { return filters_.get(); };           // returns 0 if RefProd is null
      const TriggerFilter           * filter( const std::string & labelFilter ) const;      // returns 0 if filter is not found
      unsigned                        indexFilter( const std::string & labelFilter ) const; // returns size of filter collection if filter is not found
      TriggerFilterRefVector          acceptedFilters() const;                              // transient
      
      /// objects related
      void setObjects( const edm::Handle< TriggerObjectCollection > & handleTriggerObjects ) { objects_ = TriggerObjectRefProd( handleTriggerObjects ); };
      bool addObjectMatchResult( const TriggerObjectMatchRefProd & trigMatches, const std::string & labelMatcher );         // returns 'false' if 'matcher' alreadey exists
      bool addObjectMatchResult( const edm::Handle< TriggerObjectMatch > & trigMatches, const std::string & labelMatcher ); // returns 'false' if 'matcher' alreadey exists
      const TriggerObjectCollection     * objects() const { return objects_.get(); };                                       // returns 0 if RefProd is null
      TriggerObjectRefVector              objects( unsigned filterId ) const;                                               // transient
      
      /// x-collection related
      TriggerFilterRefVector     pathModules( const std::string & namePath, bool all = true ) const;                          // transient; setting 'all' to 'false' returns the run filters only.
      TriggerFilterRefVector     pathFilters( const std::string & namePath ) const;                                           // transient; only active filter modules
      bool                       filterInPath( const TriggerFilterRef & filterRef, const std::string & namePath ) const;      // returns 'true' if the filter was run
      TriggerPathRefVector       filterPaths( const TriggerFilterRef & filterRef ) const;                                     // transient
      std::vector< std::string > filterCollections( const std::string & labelFilter ) const;                                  // returns the used collections, not the configuration
      TriggerObjectRefVector     filterObjects( const std::string & labelFilter ) const;                                      // transient
      bool                       objectInFilter( const TriggerObjectRef & objectRef, const std::string & labelFilter ) const;
      TriggerFilterRefVector     objectFilters( const TriggerObjectRef & objectRef ) const;                                   // transient
      TriggerObjectRefVector     pathObjects( const std::string & namePath ) const;                                           // transient
      bool                       objectInPath( const TriggerObjectRef & objectRef, const std::string & namePath ) const;
      TriggerPathRefVector       objectPaths( const TriggerObjectRef & objectRef  ) const;                                    // transient
      
      /// trigger matches
      std::vector< std::string >          triggerMatchers() const;
      const TriggerObjectMatchContainer * triggerObjectMatchResults() const { return &objectMatchResults_; };
      // pat::TriggerObjectMatch can contain empty references in case no match for a PAT object was found.
      const TriggerObjectMatch          * triggerObjectMatchResult( const std::string & labelMatcher ) const;                                                              // performs proper "range check" (better than '(*triggerObjectMatchResults())[labelMatcher]'), returns 0 if 'labelMatcher' not found
      // For retrieving matches for given refs, the event has to be passed as argument due to the usage of edm::AssociativeIterator
      // PAT objects do not have multiple trigger matches per matcher module
      TriggerObjectRef                    triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher, const edm::Event & iEvent ) const; // transient, returns null-Ref if no match is found
      TriggerObjectMatchMap               triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & iEvent ) const;                                  // transient
      // trigger objects can have multiple trigger matches per matcher module (resolveAmbiguities=false)
      reco::CandidateBaseRefVector        triggerMatchCandidates( const TriggerObjectRef & objectRef, const std::string & labelMatcher, const edm::Event & iEvent ) const; // transient
      
  };

}


#endif
