#ifndef L1ExtraFromDigis_L1GctInternJetProducer_h
#define L1ExtraFromDigis_L1GctInternJetProducer_h
// -*- C++ -*-
//
// Package:     EventFilter/GctRawToDigi
// Class  :     L1GctInternJetProducer
// 
/**\class L1GctInternJetProducer \file L1GctInternJetProducer.h EventFilter/GctRawToDigi/plugins/L1GctInternJetProducer.h 

\author Alex Tapper

 Description: producer of L1Extra style internal GCT jets from Level-1 hardware objects.

*/

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

// forward declarations
class L1CaloGeometry ;

class L1GctInternJetProducer : public edm::EDProducer {
   public:
      explicit L1GctInternJetProducer(const edm::ParameterSet&);
      ~L1GctInternJetProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      math::PtEtaPhiMLorentzVector gctLorentzVector( const double& et,
						     const L1GctCand& cand,
						     const L1CaloGeometry* geom,
						     bool central ) ;

      edm::InputTag internalJetSource_;

      bool centralBxOnly_;

};

#endif
