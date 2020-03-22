#ifndef ElectronRecalibSuperClusterAssociator_h
#define ElectronRecalibSuperClusterAssociator_h

//
// Package:         Calibration/EcalCalibAlgos
// Class:           ElectronRecalibSuperClusterAssociator
//
// Description:

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <string>

//class PixelMatchElectronAlgo;

class ElectronRecalibSuperClusterAssociator : public edm::EDProducer {
public:
  explicit ElectronRecalibSuperClusterAssociator(const edm::ParameterSet& conf);

  ~ElectronRecalibSuperClusterAssociator() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  edm::InputTag electronSrc_;
  edm::InputTag superClusterCollectionEB_;
  edm::InputTag superClusterCollectionEE_;

  std::string outputLabel_;

  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> ebScToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> eeScToken_;
};
#endif
