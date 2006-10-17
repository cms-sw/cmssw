#ifndef L1ExtraFromDigis_L1ExtraParticleMapProd_h
#define L1ExtraFromDigis_L1ExtraParticleMapProd_h
// -*- C++ -*-
//
// Package:     L1ExtraFromDigis
// Class  :     L1ExtraParticleMapProd
// 
/**\class L1ExtraParticleMapProd L1ExtraParticleMapProd.h L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticleMapProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Oct 17 00:14:00 EDT 2006
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// forward declarations

class L1ExtraParticleMapProd : public edm::EDProducer {
   public:
      explicit L1ExtraParticleMapProd(const edm::ParameterSet&);
      ~L1ExtraParticleMapProd();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:

      // Adds Refs to the objects in handle to the outputRefs vector.
      template< class TCollection >
      void addToVectorRefs(
	 const edm::Handle< TCollection >& handle,               // input
	 std::vector< edm::Ref< TCollection > >& vectorRefs ) ;  // output

      template< class TCollection >
      void evaluateSingleObjectTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs, // input
	 const double& etThreshold,                               // input
	 bool& decision,                                          // output
	 std::vector< edm::Ref< TCollection > >& outputRefs ) ;   // output

      template< class TCollection >
      void evaluateDoubleSameObjectTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs, // input
	 const double& etThreshold,                               // input
	 bool& decision,                                          // output
	 std::vector< edm::Ref< TCollection > >& outputRefs,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;   // output

      template< class TCollection >
      void evaluateTripleSameObjectTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs, // input
	 const double& etThreshold,                               // input
	 bool& decision,                                          // output
	 std::vector< edm::Ref< TCollection > >& outputRefs,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;   // output

      template< class TCollection >
      void evaluateQuadSameObjectTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs, // input
	 const double& etThreshold,                               // input
	 bool& decision,                                          // output
	 std::vector< edm::Ref< TCollection > >& outputRefs,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;   // output

      template< class TCollection1, class TCollection2 >
      void evaluateDoubleDifferentObjectTrigger(
	 const std::vector< edm::Ref< TCollection1 > >& inputRefs1, // input
	 const std::vector< edm::Ref< TCollection2 > >& inputRefs2, // input
	 const double& etThreshold1,                                // input
	 const double& etThreshold2,                                // input
	 bool& decision,                                            // output
	 std::vector< edm::Ref< TCollection1 > >& outputRefs1,      // output
	 std::vector< edm::Ref< TCollection2 > >& outputRefs2,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;     // output

      // ----------member data ---------------------------
      edm::InputTag isoEmSource_ ;
      edm::InputTag nonIsoEmSource_ ;
      edm::InputTag cenJetSource_ ;
      edm::InputTag forJetSource_ ;
      edm::InputTag tauJetSource_ ;
      edm::InputTag muonSource_ ;
      edm::InputTag etMissSource_ ;

      double singleIsoEmMinEt_ ;
      double doubleIsoEmMinEt_ ;

      double singleRelaxedEmMinEt_ ;
      double doubleRelaxedEmMinEt_ ;

      double singleMuonMinEt_ ;
      double doubleMuonMinEt_ ;

      double singleTauMinEt_ ;
      double doubleTauMinEt_ ;

      double singleJetMinEt_ ;
      double doubleJetMinEt_ ;
      double tripleJetMinEt_ ;
      double quadJetMinEt_ ;

      double htMin_ ;
      double metMin_ ;

      double htMetMinHt_ ;
      double htMetMinMet_ ;

      double jetMetMinJetEt_ ;
      double jetMetMinMet_ ;

      double tauMetMinTauEt_ ;
      double tauMetMinMet_ ;

      double muonMetMinMuonEt_ ;
      double muonMetMinMet_ ;

      double isoEmMetMinEmEt_ ;
      double isoEmMetMinMet_ ;

      double muonJetMinMuonEt_ ;
      double muonJetMinJetEt_ ;

      double isoEmJetMinEmEt_ ;
      double isoEmJetMinJetEt_ ;

      double muonTauMinMuonEt_ ;
      double muonTauMinTauEt_ ;

      double isoEmTauMinEmEt_ ;
      double isoEmTauMinTauEt_ ;

      double isoEmMuonMinEmEt_ ;
      double isoEmMuonMinMuonEt_ ;

      int singleJet140Prescale_ ;
      int singleJet60Prescale_ ;
      int singleJet20Prescale_ ;

      int minBiasPrescale_ ;
};

#endif
