// -*- C++ -*-


// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class decleration
//

class AlCaHcalNoiseProducer : public edm::EDProducer {
   public:
      explicit AlCaHcalNoiseProducer(const edm::ParameterSet&);
      ~AlCaHcalNoiseProducer();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 edm::InputTag JetSource_;
 edm::InputTag MetSource_;
 edm::InputTag TowerSource_;
 bool useMet_;
 bool useJet_;
 double MetCut_;
 double JetMinE_;
 double JetHCALminEnergyFraction_;
 int nAnomalousEvents;
 int nEvents;

 edm::InputTag hbheLabel_;
 edm::InputTag hoLabel_;
 edm::InputTag hfLabel_;
 std::vector<edm::InputTag> ecalLabels_;
 edm::InputTag ecalPSLabel_; 
 edm::InputTag rawInLabel_;
};
