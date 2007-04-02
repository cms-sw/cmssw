#ifndef L1ExtraFromDigis_L1ExtraParticlesProd_h
#define L1ExtraFromDigis_L1ExtraParticlesProd_h
// -*- C++ -*-
//
// Package:     L1ExtraFromDigis
// Class  :     L1ExtraParticlesProd
// 
/**\class L1ExtraParticlesProd \file L1ExtraParticlesProd.h L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticlesProd.h \author Werner Sun

 Description: producer of L1Extra particle objects from Level-1 hardware objects.

*/
//
// Original Author:  
//         Created:  Tue Oct 17 00:13:51 EDT 2006
// $Id: L1ExtraParticlesProd.h,v 1.1 2006/10/17 21:41:32 wsun Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

// forward declarations

class L1ExtraParticlesProd : public edm::EDProducer {
   public:
      enum GctBins { kNumberGctEmJetPhiBins = 18,
		     kNumberGctEtSumPhiBins = 72,
		     kNumberGctCentralEtaBinsPerHalf = 7,
		     kNumberGctForwardEtaBinsPerHalf = 4 } ;

      explicit L1ExtraParticlesProd(const edm::ParameterSet&);
      ~L1ExtraParticlesProd();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      math::XYZTLorentzVector gctLorentzVector( const double& et,
						const L1GctCand& cand,
						bool central ) ;
      
      // ----------member data ---------------------------
      bool produceMuonParticles_ ;
      edm::InputTag muonSource_ ;

      bool produceCaloParticles_ ;
      edm::InputTag isoEmSource_ ;
      edm::InputTag nonIsoEmSource_ ;
      edm::InputTag cenJetSource_ ;
      edm::InputTag forJetSource_ ;
      edm::InputTag tauJetSource_ ;
      edm::InputTag etTotSource_ ;
      edm::InputTag etHadSource_ ;
      edm::InputTag etMissSource_ ;

      static double muonMassGeV_ ;

      // Calo phi bins are uniform.
      static double gctPhiOffset_ ;
      static double gctEmJetPhiBinWidth_ ;
      static double gctEtSumPhiBinWidth_ ;

      // Calo eta bins are non-uniform.

      // Calo eta sign bit is the 4th bit.
      static unsigned gctEtaSignBitOffset_ ;

      // Extra element is for min eta.
      static double gctEtaBinBoundaries_[
	 kNumberGctCentralEtaBinsPerHalf +
	 kNumberGctForwardEtaBinsPerHalf + 1 ] ;
};

#endif
