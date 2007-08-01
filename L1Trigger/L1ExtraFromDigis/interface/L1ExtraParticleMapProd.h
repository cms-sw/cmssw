#ifndef L1ExtraFromDigis_L1ExtraParticleMapProd_h
#define L1ExtraFromDigis_L1ExtraParticleMapProd_h
// -*- C++ -*-
//
// Package:     L1ExtraFromDigis
// Class  :     L1ExtraParticleMapProd
// 
/**\class L1ExtraParticleMapProd \file L1ExtraParticleMapProd.h L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticleMapProd.h \author Werner Sun

 Description: producer of L1ParticleMap objects from GT emulator object maps.
*/
//
// Original Author:  
//         Created:  Tue Oct 17 00:14:00 EDT 2006
// $Id: L1ExtraParticleMapProd.h,v 1.6 2007/07/14 19:03:30 wsun Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
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
	 l1extra::L1ParticleMap::L1IndexComboVector& combos,      // output
	 bool combinedWithGlobalObject = false ) ; // if true, add entry for
                                             // HT or MET to particle combos

      template< class TCollection >
      void evaluateTripleSameObjectTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs, // input
	 const double& etThreshold,                               // input
	 bool& decision,                                          // output
	 std::vector< edm::Ref< TCollection > >& outputRefs,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;   // output

      template< class TCollection1, class TCollection2 >
      void evaluateDoublePlusSingleObjectTrigger(
	 const std::vector< edm::Ref< TCollection1 > >& inputRefs1, // input
	 const std::vector< edm::Ref< TCollection2 > >& inputRefs2, // input
	 const double& etThreshold1,                                // input
	 const double& etThreshold2,                                // input
	 bool& decision,                                            // output
	 std::vector< edm::Ref< TCollection1 > >& outputRefs1,      // output
	 std::vector< edm::Ref< TCollection2 > >& outputRefs2,      // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;     // output

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

      template< class TCollection >
      void evaluateDoubleDifferentObjectSameTypeTrigger(
	 const std::vector< edm::Ref< TCollection > >& inputRefs1, // input
	 const std::vector< edm::Ref< TCollection > >& inputRefs2, // input
	 const double& etThreshold1,                               // input
	 const double& etThreshold2,                               // input
	 bool& decision,                                           // output
	 std::vector< edm::Ref< TCollection > >& outputRefs,       // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;    // output

      void evaluateDoubleDifferentCaloObjectTrigger(
	 const l1extra::L1EmParticleVectorRef& inputRefs1,          // input
	 const l1extra::L1JetParticleVectorRef& inputRefs2,         // input
	 const double& etThreshold1,                                // input
	 const double& etThreshold2,                                // input
	 bool& decision,                                            // output
	 l1extra::L1EmParticleVectorRef& outputRefs1,               // output
	 l1extra::L1JetParticleVectorRef& outputRefs2,              // output
	 l1extra::L1ParticleMap::L1IndexComboVector& combos ) ;     // output

      void evaluateJetGapJetTrigger(
         const l1extra::L1JetParticleVectorRef& inputRefs,          // input
         const double& etThreshold,                                 // input
         bool& decision,                                            // output
         l1extra::L1JetParticleVectorRef& outputRefs,               // output
         l1extra::L1ParticleMap::L1IndexComboVector& combos );      // output

       void evaluateForwardRapidityGap(
          const l1extra::L1JetParticleVectorRef& inputRefs,         // input
          const double& etThreshold,                                // input
          bool& decision );                                         // output

/*       template< class TCollection1, class TCollection2 > */
/*       void evaluateDoubleObjectPlusForwardRapidityGapTrigger ( */
/* 	  const std::vector< edm::Ref< TCollection1 > >& inputRefs1,// input */
/*           const std::vector< edm::Ref< TCollection2 > >& inputRefs2,// input */
/*           const double& etThreshold1,                               // input */
/*           const double& etThreshold2,                               // input */
/*           const double& etThreshold3,                               // input */
/*           bool& decision,                                           // output */
/*           std::vector< edm::Ref< TCollection1 > >& outputRefs1,     // output */
/*           std::vector< edm::Ref< TCollection2 > >& outputRefs2,     // output */
/*           l1extra::L1ParticleMap::L1IndexComboVector& combos );     // output */

      // ----------member data ---------------------------
      edm::InputTag muonSource_ ;
      edm::InputTag isoEmSource_ ;
      edm::InputTag nonIsoEmSource_ ;
      edm::InputTag cenJetSource_ ;
      edm::InputTag forJetSource_ ;
      edm::InputTag tauJetSource_ ;
      edm::InputTag etMissSource_ ;

      double singleThresholds_[ l1extra::L1ParticleMap::kNumOfL1TriggerTypes ];
      int prescales_[ l1extra::L1ParticleMap::kNumOfL1TriggerTypes ] ;
      int prescaleCounters_[
	 l1extra::L1ParticleMap::kNumOfL1TriggerTypes ] ;
      std::pair< double, double >
         doubleThresholds_[ l1extra::L1ParticleMap::kNumOfL1TriggerTypes ] ;
};

#endif
