#ifndef DataFormats_PatCandidates_TriggerObject_h
#define DataFormats_PatCandidates_TriggerObject_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObject
//
// $Id: TriggerObject.h,v 1.6 2010/02/25 16:15:32 vadler Exp $
//
/**
  \class    pat::TriggerObject TriggerObject.h "DataFormats/PatCandidates/interface/TriggerObject.h"
  \brief    Analysis-level trigger object class

   TriggerObject implements a container for trigger objects' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerObject

  \author   Volker Adler
  \version  $Id: TriggerObject.h,v 1.6 2010/02/25 16:15:32 vadler Exp $
*/


#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include <map>
#include <string>
#include <vector>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace pat {

  class TriggerObject : public reco::LeafCandidate {

      /// data members
      std::string        collection_;
      std::vector< int > filterIds_;  // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
                                      // empty, if object was not used in last active filter

    public:

      /// constructors and desctructor
      TriggerObject();
      TriggerObject( const reco::Particle::LorentzVector & vec, int id = 0 );
      TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id = 0 );
      TriggerObject( const trigger::TriggerObject & trigObj );
      TriggerObject( const reco::LeafCandidate & leafCand );
      virtual ~TriggerObject() {};

      /// setters & getters
      void setCollection( const std::string & coll )   { collection_ = coll; };
      void setCollection( const edm::InputTag & coll ) { collection_ = coll.encode(); };
      void addFilterId( int filterId )                 { filterIds_.push_back( filterId ); };
      std::string        collection() const                                { return collection_; };
      bool               hasCollection( const edm::InputTag & coll ) const { return hasCollection( coll.encode() ); };
      bool               hasCollection( const std::string & coll ) const;
      std::vector< int > filterIds() const                                 { return filterIds_; };
      bool               hasFilterId( int filterId ) const;

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
  /// const iterator over vector of persistent references to items in the same TriggerObjectCollection
  typedef edm::RefVectorIterator< TriggerObjectCollection >  TriggerObjectRefVectorIterator;
  /// association of TriggerObjects to store matches to Candidates
  typedef edm::Association< TriggerObjectCollection >        TriggerObjectMatch;
  /// persistent reference to a TriggerObjectMatch product
  typedef edm::RefProd< TriggerObjectMatch >                 TriggerObjectMatchRefProd;
  /// container to store references to matches from different producers in the trigger event
  typedef std::map< std::string, TriggerObjectMatchRefProd > TriggerObjectMatchContainer;

}


#endif
