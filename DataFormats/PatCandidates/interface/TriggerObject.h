#ifndef DataFormats_PatCandidates_TriggerObject_h
#define DataFormats_PatCandidates_TriggerObject_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObject
//
// $Id: TriggerObject.h,v 1.8 2010/06/26 17:53:57 vadler Exp $
//
/**
  \class    pat::TriggerObject TriggerObject.h "DataFormats/PatCandidates/interface/TriggerObject.h"
  \brief    Analysis-level trigger object class

   TriggerObject implements a container for trigger objects' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerObject

  \author   Volker Adler
  \version  $Id: TriggerObject.h,v 1.8 2010/06/26 17:53:57 vadler Exp $
*/


#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include <map>
#include <string>
#include <vector>

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace pat {

  class TriggerObject : public reco::LeafCandidate {

      /// data members
      std::string                               collection_;
      std::vector< trigger::TriggerObjectType > filterIds_;  // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
                                                             // empty, if object was not used in last active filter
      reco::CandidateBaseRef refToOrig_;                     // reference to original trigger object, meant for 'l1extra' particles to access their additional functionalities

    public:

      /// constructors and desctructor
      TriggerObject();
      TriggerObject( const reco::Particle::LorentzVector & vec, int id = 0 );
      TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id = 0 );
      TriggerObject( const trigger::TriggerObject & trigObj );
      TriggerObject( const reco::LeafCandidate & leafCand );
      TriggerObject( const reco::CandidateBaseRef & candRef );
      virtual ~TriggerObject() {};

      /// setters & getters
      void setCollection( const std::string & coll )          { collection_ = coll; };
      void setCollection( const edm::InputTag & coll )        { collection_ = coll.encode(); };
      void addFilterId( trigger::TriggerObjectType filterId ) { filterIds_.push_back( filterId ); };
      void addFilterId( int filterId )                        { addFilterId( trigger::TriggerObjectType( filterId ) ); };
      std::string                               collection() const                                       { return collection_; };
      bool                                      hasCollection( const edm::InputTag & coll ) const        { return hasCollection( coll.encode() ); };
      bool                                      hasCollection( const std::string & coll ) const;
      bool                                      coll( const std::string & coll ) const                   { return hasCollection( coll );};
      std::vector< trigger::TriggerObjectType > filterIds() const                                        { return filterIds_; };
      bool                                      hasFilterId( trigger::TriggerObjectType filterId ) const;
      bool                                      hasFilterId( int filterId ) const                        { return hasFilterId( trigger::TriggerObjectType( filterId ) ); };
      bool                                      id( trigger::TriggerObjectType filterId ) const          { return hasFilterId( filterId ); };
      bool                                      id( int filterId ) const                                 { return id( trigger::TriggerObjectType ( filterId ) ); };
      // special general getters for 'l1extra' particles
      const reco::CandidateBaseRef & origObjRef()  const { return refToOrig_; };
      const reco::Candidate        * origObjCand() const { return refToOrig_.get(); };
      // special specific getters for 'l1extra' particles
      const l1extra::L1EmParticleRef       origL1EmRef()       const;
      const L1GctEmCand                  * origL1GctEmCand()   const { return origL1EmRef().isNonnull() ? origL1EmRef()->gctEmCand() : 0; };
      const l1extra::L1EtMissParticleRef   origL1EtMissRef()   const;
      const L1GctEtMiss                  * origL1GctEtMiss()   const { return origL1EtMissRef().isNonnull() ? origL1EtMissRef()->gctEtMiss() : 0; };
      const L1GctEtTotal                 * origL1GctEtTotal()  const { return origL1EtMissRef().isNonnull() ? origL1EtMissRef()->gctEtTotal() : 0; };
      const L1GctHtMiss                  * origL1GctHtMiss()   const { return origL1EtMissRef().isNonnull() ? origL1EtMissRef()->gctHtMiss() : 0; };
      const L1GctEtHad                   * origL1GctEtHad()    const { return origL1EtMissRef().isNonnull() ? origL1EtMissRef()->gctEtHad() : 0; };
      const l1extra::L1JetParticleRef      origL1JetRef()      const;
      const L1GctJetCand                 * origL1GctJetCand()  const { return origL1JetRef().isNonnull() ? origL1JetRef()->gctJetCand() : 0; };
      const l1extra::L1MuonParticleRef     origL1MuonRef()     const;
      const L1MuGMTExtendedCand          * origL1GmtMuonCand() const { return origL1MuonRef().isNonnull() ? &( origL1MuonRef()->gmtMuonCand() ) : 0; };

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
