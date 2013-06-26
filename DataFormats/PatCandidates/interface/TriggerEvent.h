#ifndef DataFormats_PatCandidates_TriggerEvent_h
#define DataFormats_PatCandidates_TriggerEvent_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerEvent
//
// $Id: TriggerEvent.h,v 1.17 2013/06/11 13:24:49 vadler Exp $
//
/**
  \class    pat::TriggerEvent TriggerEvent.h "DataFormats/PatCandidates/interface/TriggerEvent.h"
  \brief    Analysis-level trigger event class

   TriggerEvent implements a container for trigger event's information within the 'pat' namespace
   and provides the central entry point to all trigger information in the PAT.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerEvent

  \author   Volker Adler
  \version  $Id: TriggerEvent.h,v 1.17 2013/06/11 13:24:49 vadler Exp $
*/


#include "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"
#include "DataFormats/PatCandidates/interface/TriggerCondition.h"
#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include <string>
#include <vector>
#include <boost/cstdint.hpp>

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Candidate/interface/Candidate.h"


namespace pat {

  class TriggerEvent {

      /// Data Members

      /// Name of the L1 trigger menu
      std::string nameL1Menu_;
      /// Name of the HLT trigger table
      std::string nameHltTable_;
      /// Was HLT run?
      bool run_;
      /// Did HLT succeed?
      bool accept_;
      /// Was HLT in error?
      bool error_;
      /// PhysicsDeclared GT bit
      bool physDecl_;
      /// LHC fill number
      boost::uint32_t lhcFill_;
      /// LHC beam mode
      /// as defined in http://bdidev1.cern.ch/bdisoft/operational/abbdisw_wiki/LHC/BST-config --> Beam mode.
      boost::uint16_t beamMode_;
      /// LHC beam momentum in GeV
      boost::uint16_t beamMomentum_;
      /// LHC beam 1 intensity in ???
      boost::uint32_t intensityBeam1_;
      /// LHC beam 2 intensity in ???
      boost::uint32_t intensityBeam2_;
      /// LHC master status
      /// as defined in http://bdidev1.cern.ch/bdisoft/operational/abbdisw_wiki/LHC/BST-config
      boost::uint16_t bstMasterStatus_;
      /// LHC beam turn counter
      boost::uint32_t turnCount_;
      /// CMS magnet current in ??? at start of run
      float bCurrentStart_;
      /// CMS magnet current in ??? at end of run
      float bCurrentStop_;
      /// CMS magnet current in ??? averaged over run
      float bCurrentAvg_;

      /// Member collection related data members
      /// Reference to pat::TriggerAlgorithmCollection in event
      TriggerAlgorithmRefProd algorithms_;
      /// Reference to pat::TriggerConditionCollection in event
      TriggerConditionRefProd conditions_;
      /// Reference to pat::TriggerPathCollection in event
      TriggerPathRefProd paths_;
      /// Reference to pat::TriggerAlgorithmCollection in event
      TriggerFilterRefProd filters_;
      /// Reference to pat::TriggerObjectCollection in event
      TriggerObjectRefProd objects_;
      /// Table of references to pat::TriggerObjectMatch associations in event
      TriggerObjectMatchContainer objectMatchResults_;

    public:

      /// Constructors and Desctructor

      /// Default constructor
      TriggerEvent();
      /// Constructor from values, HLT only
      TriggerEvent( const std::string & nameHltTable, bool run = true, bool accept = true, bool error = false, bool physDecl = true );
      /// Constructor from values, HLT and L1/GT
      TriggerEvent( const std::string & nameL1Menu, const std::string & nameHltTable, bool run = true, bool accept = true, bool error = false, bool physDecl = true );

      /// Destructor
      virtual ~TriggerEvent() {};

      /// Methods

      /// Trigger event
      /// Set the name of the L1 trigger menu
      void setNameL1Menu( const std::string & name ) { nameL1Menu_  = name; };
      /// Set the name of the HLT trigger table
      void setNameHltTable( const std::string & name ) { nameHltTable_ = name; };
      /// Set the run flag
      void setRun( bool run ) { run_ = run; };
      /// Set the success flag
      void setAccept( bool accept ) { accept_ = accept; };
      /// Set the error flag
      void setError( bool error ) { error = error; };
      /// Set the PhysicsDeclared GT bit
      void setPhysDecl( bool physDecl ) { physDecl_ = physDecl; };
      /// Set the LHC fill number
      void setLhcFill( boost::uint32_t lhcFill )  { lhcFill_  = lhcFill; };
      /// Set the LHC beam mode
      void setBeamMode( boost::uint16_t beamMode ) { beamMode_  = beamMode; };
      /// Set the LHC beam momentum
      void setBeamMomentum( boost::uint16_t beamMomentum ) { beamMomentum_ = beamMomentum; };
      /// Set the LHC beam 1 intensity
      void setIntensityBeam1( boost::uint32_t intensityBeam1 ) { intensityBeam1_ = intensityBeam1; };
      /// Set the LHC beam 2 intensity
      void setIntensityBeam2( boost::uint32_t intensityBeam2 ) { intensityBeam2_ = intensityBeam2; };
      /// Set the LHC master status
      void setBstMasterStatus( boost::uint16_t bstMasterStatus ) { bstMasterStatus_ = bstMasterStatus; };
      /// Set the LHC beam turn counter
      void setTurnCount( boost::uint32_t turnCount ) { turnCount_ = turnCount; };
      /// Set the CMS magnet current at start of run
      void setBCurrentStart( float bCurrentStart )  { bCurrentStart_ = bCurrentStart; };
      /// Set the CMS magnet current at end of run
      void setBCurrentStop( float bCurrentStop ) { bCurrentStop_ = bCurrentStop; };
      /// Set the CMS magnet current averaged over run
      void setBCurrentAvg( float bCurrentAvg ) { bCurrentAvg_  = bCurrentAvg; };
      /// Get the name of the L1 trigger menu
      const std::string & nameL1Menu() const { return nameL1Menu_; };
      /// Get the name of the HLT trigger table
      const std::string & nameHltTable() const { return nameHltTable_; };
      /// Get the run flag
      bool wasRun() const { return run_; };
      /// Get the success flag
      bool wasAccept() const { return accept_; };
      /// Get the error flag
      bool wasError() const { return error_; };
      /// Get the PhysicsDeclared GT bit
      bool wasPhysDecl() const { return physDecl_; };
      /// Get the LHC fill number
      boost::uint32_t lhcFill() const { return lhcFill_; };
      /// Get the LHC beam mode
      boost::uint16_t beamMode() const { return beamMode_; };
      /// Get the LHC beam momentum
      boost::uint16_t beamMomentum() const { return beamMomentum_; };
      /// Get the LHC beam 1 intensity
      boost::uint32_t intensityBeam1() const { return intensityBeam1_; };
      /// Get the LHC beam 2 intensity
      boost::uint32_t intensityBeam2() const { return intensityBeam2_; };
      /// Get the LHC master status
      boost::uint16_t bstMasterStatus() const { return bstMasterStatus_; };
      /// Get the LHC beam turn counter
      boost::uint32_t turnCount() const { return turnCount_; };
      /// Get the CMS magnet current at start of run
      float bCurrentStart() const { return bCurrentStart_; };
      /// Get the CMS magnet current at end of run
      float bCurrentStop() const { return bCurrentStop_; };
      /// Get the CMS magnet current averaged over run
      float bCurrentAvg() const { return bCurrentAvg_; };

      /// L1 algorithms
      /// Set the reference to the pat::TriggerAlgorithmCollection in the event
      void setAlgorithms( const edm::Handle< TriggerAlgorithmCollection > & handleTriggerAlgorithms ) { algorithms_ = TriggerAlgorithmRefProd( handleTriggerAlgorithms ); };
      /// Get a pointer to all L1 algorithms,
      /// returns 0, if RefProd is NULL
      const TriggerAlgorithmCollection * algorithms() const { return algorithms_.get(); };
      /// Get a vector of references to all L1 algorithms,
      /// empty, if RefProd is NULL
      const TriggerAlgorithmRefVector algorithmRefs() const;
      /// Get a pointer to a certain L1 algorithm by name,
      /// returns 0, if algorithm is not found
      const TriggerAlgorithm * algorithm( const std::string & nameAlgorithm ) const;
      /// Get a reference to a certain L1 algorithm by name,
      /// NULL, if algorithm is not found
      const TriggerAlgorithmRef algorithmRef( const std::string & nameAlgorithm ) const;
      /// Get the name of a certain L1 algorithm in the event collection by bit number physics or technical (default) algorithms,
      /// returns empty string, if algorithm is not found
      std::string nameAlgorithm( const unsigned bitAlgorithm, const bool techAlgorithm = true ) const;
      /// Get the index of a certain L1 algorithm in the event collection by name,
      /// returns size of algorithm collection, if algorithm is not found
      unsigned indexAlgorithm( const std::string & nameAlgorithm ) const;
      /// Get a vector of references to all succeeding L1 algorithms
      TriggerAlgorithmRefVector acceptedAlgorithms() const;
      /// Get a vector of references to all L1 algorithms succeeding on the GTL board
      TriggerAlgorithmRefVector acceptedAlgorithmsGtl() const;
      /// Get a vector of references to all technical L1 algorithms
      TriggerAlgorithmRefVector techAlgorithms() const;
      /// Get a vector of references to all succeeding technical L1 algorithms
      TriggerAlgorithmRefVector acceptedTechAlgorithms() const;
      /// Get a vector of references to all technical L1 algorithms succeeding on the GTL board
      TriggerAlgorithmRefVector acceptedTechAlgorithmsGtl() const;
      /// Get a vector of references to all physics L1 algorithms
      TriggerAlgorithmRefVector physAlgorithms() const;
      /// Get a vector of references to all succeeding physics L1 algorithms
      TriggerAlgorithmRefVector acceptedPhysAlgorithms() const;
      /// Get a vector of references to all physics L1 algorithms succeeding on the GTL board
      TriggerAlgorithmRefVector acceptedPhysAlgorithmsGtl() const;

      /// L1 conditions
      /// Set the reference to the pat::TriggerConditionCollection in the event
      void setConditions( const edm::Handle< TriggerConditionCollection > & handleTriggerConditions ) { conditions_ = TriggerConditionRefProd( handleTriggerConditions ); };
      /// Get a pointer to all L1 condition,
      /// returns 0, if RefProd is NULL
      const TriggerConditionCollection * conditions() const { return conditions_.get(); };
      /// Get a vector of references to all L1 conditions,
      /// empty, if RefProd is NULL
      const TriggerConditionRefVector conditionRefs() const;
      /// Get a pointer to a certain L1 condition by name,
      /// returns 0, if condition is not found
      const TriggerCondition * condition( const std::string & nameCondition ) const;
      /// Get a reference to a certain L1 condition by name,
      /// NULL, if condition is not found
      const TriggerConditionRef conditionRef( const std::string & nameCondition ) const;
      /// Get the index of a certain L1 condition in the event collection by name,
      /// returns size of condition collection, if condition is not found
      unsigned indexCondition( const std::string & nameCondition ) const;
      /// Get a vector of references to all succeeding L1 condition
      TriggerConditionRefVector acceptedConditions() const;

      /// HLT paths
      /// Set the reference to the pat::TriggerPathCollection in the event
      void setPaths( const edm::Handle< TriggerPathCollection > & handleTriggerPaths ) { paths_ = TriggerPathRefProd( handleTriggerPaths ); };
      /// Get a pointer to all HLT paths,
      /// returns 0, if RefProd is NULL
      const TriggerPathCollection * paths() const { return paths_.get(); };
      /// Get a vector of references to all HLT paths,
      /// empty, if RefProd is NULL
      const TriggerPathRefVector pathRefs() const;
      /// Get a pointer to a certain HLT path by name,
      /// returns 0, if algorithm is not found
      const TriggerPath * path( const std::string & namePath ) const;
      /// Get a reference to a certain HLT path by name,
      /// NULL, if path is not found
      const TriggerPathRef pathRef( const std::string & namePath ) const;
      /// Get the index of a certain HLT path in the event collection by name,
      /// returns size of algorithm collection, if algorithm is not found
      unsigned indexPath( const std::string & namePath ) const;
      /// Get a vector of references to all succeeding HLT paths
      TriggerPathRefVector acceptedPaths() const;

      /// HLT filters
      /// Set the reference to the pat::TriggerFilterCollection in the event
      void setFilters( const edm::Handle< TriggerFilterCollection > & handleTriggerFilters ) { filters_ = TriggerFilterRefProd( handleTriggerFilters ); };
      /// Get a pointer to all HLT filters,
      /// returns 0, if RefProd is NULL
      const TriggerFilterCollection * filters() const { return filters_.get(); };
      /// Get a vector of references to all HLT filters,
      /// empty, if RefProd is NULL
      const TriggerFilterRefVector filterRefs() const;
      /// Get a pointer to a certain HLT filter by label,
      /// returns 0, if algorithm is not found
      const TriggerFilter * filter( const std::string & labelFilter ) const;
      /// Get a reference to a certain HLT filter by label,
      /// NULL, if filter is not found
      const TriggerFilterRef filterRef( const std::string & labelFilter ) const;
      /// Get the index of a certain HLT filter in the event collection by label,
      /// returns size of algorithm collection, if algorithm is not found
      unsigned indexFilter( const std::string & labelFilter ) const;
      /// Get a vector of references to all succeeding HLT filters
      TriggerFilterRefVector acceptedFilters() const;

      /// Trigger objects
      /// Set the reference to the pat::TriggerObjectCollection in the event
      void setObjects( const edm::Handle< TriggerObjectCollection > & handleTriggerObjects ) { objects_ = TriggerObjectRefProd( handleTriggerObjects ); };
      /// Get a pointer to all trigger objects,
      /// returns 0, if RefProd is NULL
      const TriggerObjectCollection * objects() const { return objects_.get(); };
      /// Get a vector of references to all trigger objects,
      /// empty, if RefProd is NULL
      const TriggerObjectRefVector objectRefs() const;
      /// Get a vector of references to all trigger objects by trigger object type
      TriggerObjectRefVector objects( trigger::TriggerObjectType triggerObjectType ) const;
      TriggerObjectRefVector objects( int                        triggerObjectType ) const { return objects( trigger::TriggerObjectType( triggerObjectType ) ); }; // for backward compatibility

      /// L1 x-links
      /// Get a vector of references to all conditions assigned to a certain algorithm given by name
      TriggerConditionRefVector algorithmConditions( const std::string & nameAlgorithm ) const;
      /// Checks, if a condition is assigned to a certain algorithm given by name
      bool conditionInAlgorithm( const TriggerConditionRef & conditionRef, const std::string & nameAlgorithm ) const;
      /// Get a vector of references to all algorithms, which have a certain condition assigned
      TriggerAlgorithmRefVector conditionAlgorithms( const TriggerConditionRef & conditionRef ) const;
      /// Get a list of all trigger object collections used in a certain condition given by name
      std::vector< std::string > conditionCollections( const std::string & nameAlgorithm ) const;
      /// Get a vector of references to all objects, which were used in a certain condition given by name
      TriggerObjectRefVector conditionObjects( const std::string & nameCondition ) const;
      /// Checks, if an object was used in a certain condition given by name
      bool objectInCondition( const TriggerObjectRef & objectRef, const std::string & nameCondition ) const;
      /// Get a vector of references to all conditions, which have a certain object assigned
      TriggerConditionRefVector objectConditions( const TriggerObjectRef & objectRef ) const;
      /// Get a vector of references to all objects, which were used in a certain algorithm given by name
      TriggerObjectRefVector algorithmObjects( const std::string & nameAlgorithm ) const;
      /// Checks, if an object was used in a certain algorithm given by name
      bool objectInAlgorithm( const TriggerObjectRef & objectRef, const std::string & nameAlgorithm ) const;
      /// Get a vector of references to all algorithms, which have a certain object assigned
      TriggerAlgorithmRefVector objectAlgorithms( const TriggerObjectRef & objectRef  ) const;

      /// HLT x-links
      /// Get a vector of references to all modules assigned to a certain path given by name,
      /// setting 'all' to 'false' returns the run filters only.
      TriggerFilterRefVector pathModules( const std::string & namePath, bool all = true ) const;
      /// Get a vector of references to all active HLT filters assigned to a certain path given by name
      TriggerFilterRefVector pathFilters( const std::string & namePath, bool firing = true ) const;
      /// Checks, if a filter is assigned to and was run in a certain path given by name
      bool filterInPath( const TriggerFilterRef & filterRef, const std::string & namePath, bool firing = true ) const;
      /// Get a vector of references to all paths, which have a certain filter assigned
      TriggerPathRefVector filterPaths( const TriggerFilterRef & filterRef, bool firing = true ) const;
      /// Get a list of all trigger object collections used in a certain filter given by name
      std::vector< std::string > filterCollections( const std::string & labelFilter ) const;
      /// Get a vector of references to all objects, which were used in a certain filter given by name
      TriggerObjectRefVector filterObjects( const std::string & labelFilter ) const;
      /// Checks, if an object was used in a certain filter given by name
      bool objectInFilter( const TriggerObjectRef & objectRef, const std::string & labelFilter ) const;
      /// Get a vector of references to all filters, which have a certain object assigned
      TriggerFilterRefVector objectFilters( const TriggerObjectRef & objectRef, bool firing = true ) const;
      /// Get a vector of references to all objects, which were used in a certain path given by name
      TriggerObjectRefVector pathObjects( const std::string & namePath, bool firing = true ) const;
      /// Checks, if an object was used in a certain path given by name
      bool objectInPath( const TriggerObjectRef & objectRef, const std::string & namePath, bool firing = true ) const;
      /// Get a vector of references to all paths, which have a certain object assigned
      TriggerPathRefVector objectPaths( const TriggerObjectRef & objectRef, bool firing = true  ) const;

      /// Add a pat::TriggerObjectMatch association
      /// returns 'false', if 'matcher' alreadey exists
      bool addObjectMatchResult( const TriggerObjectMatchRefProd               & trigMatches, const std::string & labelMatcher );
      bool addObjectMatchResult( const edm::Handle< TriggerObjectMatch >       & trigMatches, const std::string & labelMatcher ) { return addObjectMatchResult( TriggerObjectMatchRefProd( trigMatches ), labelMatcher ); };
      bool addObjectMatchResult( const edm::OrphanHandle< TriggerObjectMatch > & trigMatches, const std::string & labelMatcher ) { return addObjectMatchResult( TriggerObjectMatchRefProd( trigMatches ), labelMatcher ); };
      /// Get a list of all linked trigger matches
      std::vector< std::string > triggerMatchers() const;
      /// Get all trigger matches
      const TriggerObjectMatchContainer * triggerObjectMatchResults() const { return &objectMatchResults_; };
      /// Get a pointer to a certain trigger match given by label,
      /// performs proper "range check" (better than '(*triggerObjectMatchResults())[labelMatcher]'),
      /// returns 0, if matcher not found
      const TriggerObjectMatch * triggerObjectMatchResult( const std::string & labelMatcher ) const;

      /// Further methods are provided by the pat::helper::TriggerMatchHelper in PhysicsTools/PatUtils/interface/TriggerHelper.h

  };

}


#endif
