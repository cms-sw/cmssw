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

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

//
// class decleration
//

class AlCaHcalNoiseProducer : public edm::EDProducer {
public:
  explicit AlCaHcalNoiseProducer(const edm::ParameterSet &);
  ~AlCaHcalNoiseProducer() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  // ----------member data ---------------------------

  bool useMet_;
  bool useJet_;
  double MetCut_;
  double JetMinE_;
  double JetHCALminEnergyFraction_;
  int nAnomalousEvents;
  int nEvents;

  std::vector<edm::InputTag> ecalLabels_;

  edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
  edm::EDGetTokenT<reco::CaloMETCollection> tok_met_;
  edm::EDGetTokenT<CaloTowerCollection> tok_tower_;

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

  edm::EDGetTokenT<EcalRecHitCollection> tok_ps_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  std::vector<edm::EDGetTokenT<EcalRecHitCollection> > toks_ecal_;
};
