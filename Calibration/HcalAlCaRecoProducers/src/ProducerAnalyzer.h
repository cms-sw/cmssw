#ifndef CalibrationHcalAlCaRecoProducersProducerAnalyzer_h
#define CalibrationHcalAlCaRecoProducersProducerAnalyzer_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// collections
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariableCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

namespace cms {

  //
  // class declaration
  //

  class ProducerAnalyzer : public edm::EDAnalyzer {
  public:
    explicit ProducerAnalyzer(const edm::ParameterSet&);
    ~ProducerAnalyzer() override;

    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void beginJob() override;
    void endJob() override;

  private:
    // ----------member data ---------------------------
    std::string nameProd_;
    std::string jetCalo_;
    std::string gammaClus_;
    std::string ecalInput_;
    std::string hbheInput_;
    std::string hoInput_;
    std::string hfInput_;
    std::string Tracks_;

    edm::EDGetTokenT<HOCalibVariableCollection> tok_hovar_;
    edm::EDGetTokenT<HORecHitCollection> tok_horeco_;
    edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    edm::EDGetTokenT<HORecHitCollection> tok_hoProd_;

    edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

    edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
    edm::EDGetTokenT<reco::SuperClusterCollection> tok_gamma_;
    edm::EDGetTokenT<reco::MuonCollection> tok_muons_;
    edm::EDGetTokenT<EcalRecHitCollection> tok_ecal_;
    edm::EDGetTokenT<reco::TrackCollection> tok_tracks_;

    edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    edm::EDGetTokenT<HBHERecHitCollection> tok_hbheProd_;

    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  };
}  // end namespace cms
#endif
