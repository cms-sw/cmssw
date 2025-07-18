// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files

#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace cms {

  //
  // class declaration
  //

  class ProducerAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit ProducerAnalyzer(const edm::ParameterSet&);
    ~ProducerAnalyzer() override = default;

    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void beginJob() override {}
    void endJob() override {}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // ----------member data ---------------------------
    std::string nameProd_;
    std::string jetCalo_;
    std::string gammaClus_;
    std::string ecalInput_;
    std::string hbheInput_;
    std::string hoInput_;
    std::string hfInput_;
    std::string tracks_;

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

using namespace reco;

namespace cms {

  //
  // constructors and destructor
  //
  ProducerAnalyzer::ProducerAnalyzer(const edm::ParameterSet& iConfig) {
    // get name of output file with histogramms

    nameProd_ = iConfig.getUntrackedParameter<std::string>("nameProd");
    jetCalo_ = iConfig.getUntrackedParameter<std::string>("jetCalo", "GammaJetJetBackToBackCollection");
    gammaClus_ = iConfig.getUntrackedParameter<std::string>("gammaClus");
    ecalInput_ = iConfig.getUntrackedParameter<std::string>("ecalInput");
    hbheInput_ = iConfig.getUntrackedParameter<std::string>("hbheInput");
    hoInput_ = iConfig.getUntrackedParameter<std::string>("hoInput");
    hfInput_ = iConfig.getUntrackedParameter<std::string>("hfInput");
    tracks_ = iConfig.getUntrackedParameter<std::string>("Tracks");

    tok_hovar_ = consumes<HOCalibVariableCollection>(edm::InputTag(nameProd_, hoInput_));
    tok_horeco_ = consumes<HORecHitCollection>(edm::InputTag("horeco"));
    tok_ho_ = consumes<HORecHitCollection>(edm::InputTag(hoInput_));
    tok_hoProd_ = consumes<HORecHitCollection>(edm::InputTag(nameProd_, hoInput_));

    tok_hf_ = consumes<HFRecHitCollection>(edm::InputTag(hfInput_));

    tok_jets_ = consumes<reco::CaloJetCollection>(edm::InputTag(nameProd_, jetCalo_));
    tok_gamma_ = consumes<reco::SuperClusterCollection>(edm::InputTag(nameProd_, gammaClus_));
    tok_muons_ = consumes<reco::MuonCollection>(edm::InputTag(nameProd_, "SelectedMuons"));
    tok_ecal_ = consumes<EcalRecHitCollection>(edm::InputTag(nameProd_, ecalInput_));
    tok_tracks_ = consumes<reco::TrackCollection>(edm::InputTag(nameProd_, tracks_));

    tok_hbheProd_ = consumes<HBHERecHitCollection>(edm::InputTag(nameProd_, hbheInput_));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag(hbheInput_));

    tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void ProducerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;

    const CaloGeometry* geo = &(iSetup.getData(tok_geom_));

    std::vector<StableProvenance const*> theProvenance;
    iEvent.getAllStableProvenance(theProvenance);
    for (auto const& provenance : theProvenance) {
      edm::LogVerbatim("HcalAlCa") << " Print all label names " << provenance->moduleLabel() << " "
                                   << provenance->productInstanceName();
    }

    if (nameProd_ == "hoCalibProducer") {
      auto const& ho = iEvent.getHandle(tok_hovar_);
      const HOCalibVariableCollection Hitho = *(ho.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HO " << (Hitho).size();
    }

    if (nameProd_ == "ALCARECOMuAlZMuMu") {
      auto const& ho = iEvent.getHandle(tok_horeco_);
      const HORecHitCollection Hitho = *(ho.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HO " << (Hitho).size();
      auto const& mucand = iEvent.getHandle(tok_muons_);
      edm::LogVerbatim("HcalAlCa") << " Size of muon collection " << mucand->size();
      for (const auto& it : *(mucand.product())) {
        TrackRef mu = it.combinedMuon();
        edm::LogVerbatim("HcalAlCa") << " Pt muon " << mu->innerMomentum();
      }
    }

    if (nameProd_ != "IsoProd" && nameProd_ != "ALCARECOMuAlZMuMu" && nameProd_ != "hoCalibProducer") {
      auto const& hbhe = iEvent.getHandle(tok_hbhe_);
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HBHE " << (Hithbhe).size();

      auto const& ho = iEvent.getHandle(tok_ho_);
      const HORecHitCollection Hitho = *(ho.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HO " << (Hitho).size();

      auto const& hf = iEvent.getHandle(tok_hf_);
      const HFRecHitCollection Hithf = *(hf.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HF " << (Hithf).size();
    }
    if (nameProd_ == "IsoProd") {
      edm::LogVerbatim("HcalAlCa") << " We are here ";
      auto const& tracks = iEvent.getHandle(tok_tracks_);

      edm::LogVerbatim("HcalAlCa") << " Tracks size " << (*tracks).size();
      for (const auto& track : *(tracks.product())) {
        edm::LogVerbatim("HcalAlCa") << " P track " << track.p() << " eta " << track.eta() << " phi " << track.phi()
                                     << " Outer " << track.outerMomentum() << " " << track.outerPosition();
        const TrackExtraRef& myextra = track.extra();
        edm::LogVerbatim("HcalAlCa") << " Track extra " << myextra->outerMomentum() << " " << myextra->outerPosition();
      }

      auto const& ecal = iEvent.getHandle(tok_ecal_);
      const EcalRecHitCollection Hitecal = *(ecal.product());
      edm::LogVerbatim("HcalAlCa") << " Size of Ecal " << (Hitecal).size();

      double energyECAL = 0.;
      double energyHCAL = 0.;

      for (const auto& hite : *(ecal.product())) {
        const GlobalPoint& posE = geo->getPosition(hite.detid());

        edm::LogVerbatim("HcalAlCa") << " Energy ECAL " << hite.energy() << " eta " << posE.eta() << " phi "
                                     << posE.phi();

        energyECAL = energyECAL + hite.energy();
      }

      auto const& hbhe = iEvent.getHandle(tok_hbheProd_);
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
      edm::LogVerbatim("HcalAlCa") << " Size of HBHE " << (Hithbhe).size();

      for (const auto& hith : *(hbhe.product())) {
        GlobalPoint posH =
            (static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(hith.detid())))->getPosition(hith.detid());

        edm::LogVerbatim("HcalAlCa") << " Energy HCAL " << hith.energy() << " eta " << posH.eta() << " phi "
                                     << posH.phi();

        energyHCAL = energyHCAL + hith.energy();
      }

      edm::LogVerbatim("HcalAlCa") << " Energy ECAL " << energyECAL << " Energy HCAL " << energyHCAL;
    }

    if (nameProd_ == "GammaJetProd" || nameProd_ == "DiJProd") {
      edm::LogVerbatim("HcalAlCa") << " we are in GammaJetProd area ";
      auto const& ecal = iEvent.getHandle(tok_ecal_);
      edm::LogVerbatim("HcalAlCa") << " Size of ECAL " << (*ecal).size();

      auto const& jets = iEvent.getHandle(tok_jets_);
      edm::LogVerbatim("HcalAlCa") << " Jet size " << (*jets).size();

      for (const auto& jet : *(jets.product())) {
        edm::LogVerbatim("HcalAlCa") << " Et jet " << jet.et() << " eta " << jet.eta() << " phi " << jet.phi();
      }

      auto const& tracks = iEvent.getHandle(tok_tracks_);
      edm::LogVerbatim("HcalAlCa") << " Tracks size " << (*tracks).size();
    }
    if (nameProd_ == "GammaJetProd") {
      auto const& eclus = iEvent.getHandle(tok_gamma_);
      edm::LogVerbatim("HcalAlCa") << " GammaClus size " << (*eclus).size();
      for (const auto& iclus : *(eclus.product())) {
        edm::LogVerbatim("HcalAlCa") << " Et gamma " << iclus.energy() / cosh(iclus.eta()) << " eta " << iclus.eta()
                                     << " phi " << iclus.phi();
      }
    }
  }

  void ProducerAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::string>("nameProd", "hoCalibProducer");
    desc.addUntracked<std::string>("jetCalo", "GammaJetJetBackToBackCollection");
    desc.addUntracked<std::string>("gammaClus", "GammaJetGammaBackToBackCollection");
    desc.addUntracked<std::string>("ecalInput", "GammaJetEcalRecHitCollection");
    desc.addUntracked<std::string>("hbheInput", "hbhereco");
    desc.addUntracked<std::string>("hoInput", "horeco");
    desc.addUntracked<std::string>("hfInput", "hfreco");
    desc.addUntracked<std::string>("Tracks", "GammaJetTracksCollection");
    descriptions.add("alcaHcalProducerAnalyzer", desc);
  }

}  // namespace cms

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using cms::ProducerAnalyzer;
DEFINE_FWK_MODULE(ProducerAnalyzer);
