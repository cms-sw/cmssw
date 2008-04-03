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
// $Id: L1ExtraParticlesProd.h,v 1.6 2007/12/18 03:31:12 wsun Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

// forward declarations
class L1CaloGeometry ;

class L1ExtraParticlesProd : public edm::EDProducer {
   public:
      explicit L1ExtraParticlesProd(const edm::ParameterSet&);
      ~L1ExtraParticlesProd();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //      math::XYZTLorentzVector gctLorentzVector( const double& et,
      math::PtEtaPhiMLorentzVector gctLorentzVector( const double& et,
						     const L1GctCand& cand,
						     const L1CaloGeometry* geom,
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

      bool centralBxOnly_ ;
};

#endif
