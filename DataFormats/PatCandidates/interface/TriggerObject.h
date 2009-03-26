#ifndef DataFormats_PatCandidates_TriggerObject_h
#define DataFormats_PatCandidates_TriggerObject_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObject
//
// $Id: TriggerObject.h,v 1.1.2.13 2009/03/13 12:10:35 vadler Exp $
//
/**
  \class    pat::TriggerObject TriggerObject.h "DataFormats/PatCandidates/interface/TriggerObject.h"
  \brief    Analysis-level trigger object class

   TriggerObject implements a container for trigger objects' information within the 'pat' namespace.
   It inherits from LeafCandidate and adds the following data members:
   - [to be filled]
   In addition, the data member reco::Particle::pdgId_ (inherited via reco::LeafCandidate) is used
   to store the trigger object id from trigger::TriggerObject::id_.

  \author   Volker Adler
  \version  $Id: TriggerObject.h,v 1.1.2.13 2009/03/13 12:10:35 vadler Exp $
*/


#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include <map>
#include <string>
#include <vector>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"


namespace pat {
 
  class TriggerObject : public reco::LeafCandidate {
  
      /// friends to have access to the private methods dealing with transient data members
      template< class T1, class T2 > friend class PATTriggerMatchSelector;
//       friend class PATTriggerMatchSelector< reco::CandidateView::value_type, std::vector< TriggerObject >::value_type >;
      friend class PATTriggerProducer;
    
      /// data members
      std::string             collection_;
      std::vector< unsigned > filterIds_;  // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
                                           // empty, if object was not used in last active filter
                                           
      /// transient data members
      std::vector< std::string > filterLabels_; // used for trigger match definition
      std::vector< std::string > pathNames_;    // used for trigger match definition
      
      /// transient methods
      void addFilterLabel( const std::string & filterLabel ) { if ( ! hasFilterLabel( filterLabel ) ) filterLabels_.push_back( filterLabel ); };
      void addPathName( const std::string & pathName )       { if ( ! hasPathName( pathName ) )       pathNames_.push_back( pathName ); };
      std::vector< std::string > filterLabels() const { return filterLabels_; };
      std::vector< std::string > pathNames() const    { return pathNames_; };
      bool                       hasFilterLabel( const std::string & filterLabel ) const;
      bool                       hasPathName( const std::string & pathName ) const;

    public:

      /// constructors and desctructor
      TriggerObject();
      TriggerObject( const reco::Particle::LorentzVector & vec, int id = 0 );
      TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id = 0 );
      TriggerObject( const trigger::TriggerObject & aTrigObj );
      virtual ~TriggerObject() {};
      
      /// setters & getters
      void setCollection( const std::string & collection ) { collection_ = collection; };
      void addFilterId( unsigned filterId )                { filterIds_.push_back( filterId ); };
      std::string             collection() const { return collection_; };
      std::vector< unsigned > filterIds() const  { return filterIds_; };
      bool                    hasFilterId( unsigned filterId ) const;

  };
  

  /// collection of TriggerObject
  typedef std::vector< TriggerObject >                       TriggerObjectCollection;
  /// persistent reference to an item in a TriggerObjectCollection
  typedef edm::Ref< TriggerObjectCollection >                TriggerObjectRef;
  /// container to store match references from different producers (for one PAT object)
  typedef std::map< std::string, TriggerObjectRef >          TriggerObjectMatchMap;
  /// persistent reference to a TriggerObjectCollection product
  typedef edm::RefProd< TriggerObjectCollection >            TriggerObjectRefProd;
  /// vector of persistent references to items in the same TriggerObjectCollection
  typedef edm::RefVector< TriggerObjectCollection >          TriggerObjectRefVector;
  /// association of TriggerObjects to store matches to Candidates
  typedef edm::Association< TriggerObjectCollection >        TriggerObjectMatch;
  /// persistent reference to a TriggerObjectMatch product
  typedef edm::RefProd< TriggerObjectMatch >                 TriggerObjectMatchRefProd;
  /// container to store references to matches from different producers in the trigger event
  typedef std::map< std::string, TriggerObjectMatchRefProd > TriggerObjectMatchContainer;

}


#endif
