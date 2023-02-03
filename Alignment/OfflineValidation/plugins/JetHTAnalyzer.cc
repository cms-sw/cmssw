// -*- C++ -*-
//
// Package:    TkAlTools/JetHTAnalyzer
// Class:      JetHTAnalyzer
//
/**\class JetHTAnalyzer JetHTAnalyzer.cc Alignment/OfflineValidation/plugins/JetHTAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Fri, 29 Mar 2019 14:54:59 GMT
//
//      Updated by:  Jussi Viinikainen
//

// system include files
#include <algorithm>  // std::sort
#include <memory>
#include <string>
#include <vector>  // std::vector

// CMSSW includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "Alignment/OfflineValidation/interface/SmartSelectionMonitor.h"
#include "DataFormats/Common/interface/TriggerResults.h"  // Classes needed to print trigger results
#include "FWCore/Common/interface/TriggerNames.h"

//
// class declaration
//
using reco::TrackCollection;

class JetHTAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit JetHTAnalyzer(const edm::ParameterSet&);
  ~JetHTAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static bool mysorter(reco::Track i, reco::Track j) { return (i.pt() > j.pt()); }

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::InputTag pvsTag_;
  const edm::EDGetTokenT<reco::VertexCollection> pvsToken_;

  const edm::InputTag tracksTag_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

  const edm::InputTag triggerTag_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerToken_;

  const int printTriggerTable_;
  const double minVtxNdf_;
  const double minVtxWgt_;

  const std::vector<double> profilePtBorders_;
  const std::vector<int> iovList_;

  // output histograms
  edm::Service<TFileService> outfile_;
  TH1F* h_ntrks;
  TH1F* h_probePt;
  TH1F* h_probeEta;
  TH1F* h_probePhi;
  TH1F* h_probeDxy;
  TH1F* h_probeDz;
  TH1F* h_probeDxyErr;
  TH1F* h_probeDzErr;

  SmartSelectionMonitor mon;

  // for the conversions
  static constexpr double cmToum = 10000;
};

//
// Constructor
//
JetHTAnalyzer::JetHTAnalyzer(const edm::ParameterSet& iConfig)
    : pvsTag_(iConfig.getParameter<edm::InputTag>("vtxCollection")),
      pvsToken_(consumes<reco::VertexCollection>(pvsTag_)),
      tracksTag_(iConfig.getParameter<edm::InputTag>("trackCollection")),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      triggerTag_(iConfig.getParameter<edm::InputTag>("triggerResults")),
      triggerToken_(consumes<edm::TriggerResults>(triggerTag_)),
      printTriggerTable_(iConfig.getUntrackedParameter<int>("printTriggerTable")),
      minVtxNdf_(iConfig.getUntrackedParameter<double>("minVertexNdf")),
      minVtxWgt_(iConfig.getUntrackedParameter<double>("minVertexMeanWeight")),
      profilePtBorders_(iConfig.getUntrackedParameter<std::vector<double>>("profilePtBorders")),
      iovList_(iConfig.getUntrackedParameter<std::vector<int>>("iovList")) {
  // Specify that TFileService is used by the class
  usesResource(TFileService::kSharedResource);
}

//
// member functions
//

// ------------ method called for each event  ------------
void JetHTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto& vertices = iEvent.get(pvsToken_);
  const reco::VertexCollection& pvtx = vertices;

  edm::Handle<reco::TrackCollection> tracks = iEvent.getHandle(tracksToken_);

  // Find the IOV of the current event so that we can tag a histogram with the IOV
  const int runNumber = iEvent.id().run();
  std::string iovString = "iovNotFound";

  // Find the current IOV from the IOV list
  if (runNumber >= iovList_.at(0)) {  // If run number is smaller than the first item in the list, it is not in any IOV
    for (std::vector<int>::size_type i = 1; i < iovList_.size(); i++) {
      if (iovList_.at(i) > runNumber) {
        iovString = Form("iov%d-%d", iovList_.at(i - 1), iovList_.at(i) - 1);
        break;
      }
    }
  }

  // Print the triggers to console
  if (printTriggerTable_) {
    const auto& triggerResults = iEvent.get(triggerToken_);

    const edm::TriggerNames& triggerNames = iEvent.triggerNames(triggerResults);
    for (unsigned i = 0; i < triggerNames.size(); i++) {
      const std::string& hltName = triggerNames.triggerName(i);
      bool decision = triggerResults.accept(triggerNames.triggerIndex(hltName));
      std::cout << hltName << " " << decision << std::endl;
    }
  }

  int counter = 0;
  for (reco::VertexCollection::const_iterator pvIt = pvtx.begin(); pvIt != pvtx.end(); pvIt++) {
    const reco::Vertex& iPV = *pvIt;
    counter++;

    if (iPV.isFake())
      continue;
    reco::Vertex::trackRef_iterator trki;

    const math::XYZPoint pos_(iPV.x(), iPV.y(), iPV.z());

    // vertex selection as in bs code
    if (iPV.ndof() < minVtxNdf_ || (iPV.ndof() + 3.) / iPV.tracksSize() < 2 * minVtxWgt_)
      continue;

    reco::TrackCollection allTracks;
    for (trki = iPV.tracks_begin(); trki != iPV.tracks_end(); ++trki) {
      if (trki->isNonnull()) {
        reco::TrackRef trk_now(tracks, (*trki).key());
        allTracks.push_back(*trk_now);
      }
    }

    // order with decreasing pt
    std::sort(allTracks.begin(), allTracks.end(), mysorter);
    uint ntrks = allTracks.size();
    h_ntrks->Fill(ntrks);

    for (uint tracksIt = 0; tracksIt < ntrks; tracksIt++) {
      auto tk = allTracks.at(tracksIt);

      double dxyRes = tk.dxy(pos_) * cmToum;
      double dzRes = tk.dz(pos_) * cmToum;

      double dxy_err = tk.dxyError() * cmToum;
      double dz_err = tk.dzError() * cmToum;

      float trackphi = tk.phi();
      float tracketa = tk.eta();
      float trackpt = tk.pt();

      h_probePt->Fill(trackpt);
      h_probeEta->Fill(tracketa);
      h_probePhi->Fill(trackphi);

      h_probeDxy->Fill(dxyRes);
      h_probeDz->Fill(dzRes);
      h_probeDxyErr->Fill(dxy_err);
      h_probeDzErr->Fill(dz_err);

      mon.fillHisto("dxy", "all", dxyRes, 1.);
      mon.fillHisto("dz", "all", dzRes, 1.);
      mon.fillHisto("dxyerr", "all", dxy_err, 1.);
      mon.fillHisto("dzerr", "all", dz_err, 1.);

      mon.fillProfile("dxyErrVsPt", "all", trackpt, dxy_err, 1.);
      mon.fillProfile("dzErrVsPt", "all", trackpt, dz_err, 1.);

      mon.fillProfile("dxyErrVsPhi", "all", trackphi, dxy_err, 1.);
      mon.fillProfile("dzErrVsPhi", "all", trackphi, dz_err, 1.);

      mon.fillProfile("dxyErrVsEta", "all", tracketa, dxy_err, 1.);
      mon.fillProfile("dzErrVsEta", "all", tracketa, dz_err, 1.);

      // Integrated pT bins
      for (std::vector<double>::size_type i = 0; i < profilePtBorders_.size(); i++) {
        if (trackpt < profilePtBorders_.at(i))
          break;
        mon.fillProfile("dxyErrVsPtWide", "all", i, dxy_err, 1.);
        mon.fillProfile("dzErrVsPtWide", "all", i, dz_err, 1.);
      }

      // Fill IOV specific histograms
      mon.fillHisto("dxy", iovString, dxyRes, 1.);
      mon.fillHisto("dz", iovString, dzRes, 1.);
      mon.fillHisto("dxyerr", iovString, dxy_err, 1.);
      mon.fillHisto("dzerr", iovString, dz_err, 1.);

      mon.fillProfile("dxyErrVsPt", iovString, trackpt, dxy_err, 1.);
      mon.fillProfile("dzErrVsPt", iovString, trackpt, dz_err, 1.);

      mon.fillProfile("dxyErrVsPhi", iovString, trackphi, dxy_err, 1.);
      mon.fillProfile("dzErrVsPhi", iovString, trackphi, dz_err, 1.);

      mon.fillProfile("dxyErrVsEta", iovString, tracketa, dxy_err, 1.);
      mon.fillProfile("dzErrVsEta", iovString, tracketa, dz_err, 1.);

      // Integrated pT bins
      for (std::vector<double>::size_type i = 0; i < profilePtBorders_.size(); i++) {
        if (trackpt < profilePtBorders_.at(i))
          break;
        mon.fillProfile("dxyErrVsPtWide", iovString, i, dxy_err, 1.);
        mon.fillProfile("dzErrVsPtWide", iovString, i, dz_err, 1.);
      }

      if (std::abs(tracketa) < 1.) {
        mon.fillHisto("dxy", "central", dxyRes, 1.);
        mon.fillHisto("dz", "central", dzRes, 1.);
        mon.fillHisto("dxyerr", "central", dxy_err, 1.);
        mon.fillHisto("dzerr", "central", dz_err, 1.);

        mon.fillProfile("dxyErrVsPt", "central", trackpt, dxy_err, 1.);
        mon.fillProfile("dzErrVsPt", "central", trackpt, dz_err, 1.);

        mon.fillProfile("dxyErrVsPhi", "central", trackphi, dxy_err, 1.);
        mon.fillProfile("dzErrVsPhi", "central", trackphi, dz_err, 1.);

        // Integrated pT bins
        for (std::vector<double>::size_type i = 0; i < profilePtBorders_.size(); i++) {
          if (trackpt < profilePtBorders_.at(i))
            break;
          mon.fillProfile("dxyErrVsPtWide", "central", i, dxy_err, 1.);
          mon.fillProfile("dzErrVsPtWide", "central", i, dz_err, 1.);
        }
      }

    }  // loop on tracks in vertex
  }    // loop on vertices

  mon.fillHisto("nvtx", "all", counter, 1.);
}

// ------------ method called once each job just before starting event loop  ------------
void JetHTAnalyzer::beginJob() {
  h_ntrks = outfile_->make<TH1F>("h_ntrks", "n. trks;n. of tracks/vertex;n. vertices", 100, 0, 100);
  h_probePt = outfile_->make<TH1F>("h_probePt", "p_{T} of probe track;track p_{T} (GeV); tracks", 100, 0., 500.);
  h_probeEta = outfile_->make<TH1F>("h_probeEta", "#eta of the probe track;track #eta;tracks", 54, -2.8, 2.8);
  h_probePhi = outfile_->make<TH1F>("h_probePhi", "#phi of probe track;track #phi (rad);tracks", 100, -3.15, 3.15);

  h_probeDxy =
      outfile_->make<TH1F>("h_probeDxy", "d_{xy}(PV) of the probe track;track d_{xy}(PV);tracks", 200, -100, 100);
  h_probeDz = outfile_->make<TH1F>("h_probeDz", "d_{z}(PV) of the probe track;track d_{z}(PV);tracks", 200, -100, 100);
  h_probeDxyErr = outfile_->make<TH1F>(
      "h_probeDxyErr", "error on d_{xy}(PV) of the probe track;track error on d_{xy}(PV);tracks", 100, 0., 100);
  h_probeDzErr = outfile_->make<TH1F>(
      "h_probeDzErr", "error on d_{z}(PV)  of the probe track;track error on d_{z}(PV);tracks", 100, 0., 100);

  mon.addHistogram(new TH1F("nvtx", ";Vertices;Events", 50, 0, 50));
  mon.addHistogram(new TH1F("dxy", ";d_{xy};tracks", 100, -100, 100));
  mon.addHistogram(new TH1F("dz", ";d_{z};tracks", 100, -100, 100));
  mon.addHistogram(new TH1F("dxyerr", ";d_{xy} error;tracks", 100, 0., 200));
  mon.addHistogram(new TH1F("dzerr", ";d_{z} error;tracks", 100, 0., 200));
  mon.addHistogram(new TProfile("dxyErrVsPt", ";track p_{T};d_{xy} error", 100, 0., 200, 0., 100.));
  mon.addHistogram(new TProfile("dzErrVsPt", ";track p_{T};d_{z} error", 100, 0., 200, 0., 100.));
  mon.addHistogram(new TProfile("dxyErrVsPhi", ";track #varphi;d_{xy} error", 100, -M_PI, M_PI, 0., 100.));
  mon.addHistogram(new TProfile("dzErrVsPhi", ";track #varphi;d_{z} error", 100, -M_PI, M_PI, 0., 100.));
  mon.addHistogram(new TProfile("dxyErrVsEta", ";track #eta;d_{xy} error", 100, -2.5, 2.5, 0., 100.));
  mon.addHistogram(new TProfile("dzErrVsEta", ";track #eta;d_{z} error", 100, -2.5, 2.5, 0., 100.));

  // Variable size histogram depending on the given pT bin borders
  int nBins = profilePtBorders_.size();
  mon.addHistogram(
      new TProfile("dxyErrVsPtWide", ";track p_{T} wide bin;d_{xy} error", nBins, -0.5, nBins - 0.5, 0.0, 100.0));
  mon.addHistogram(
      new TProfile("dzErrVsPtWide", ";track p_{T} wide bin;d_{z} error", nBins, -0.5, nBins - 0.5, 0.0, 100.0));
}

// ------------ method called once each job just after ending the event loop  ------------
void JetHTAnalyzer::endJob() { mon.Write(); }

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void JetHTAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("JetHT validation analyzer plugin.");
  desc.add<edm::InputTag>("vtxCollection", edm::InputTag("offlinePrimaryVerticesFromRefittedTrks"));
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("trackCollection", edm::InputTag("TrackRefitter"));
  desc.addUntracked<int>("printTriggerTable", false);
  desc.addUntracked<double>("minVertexNdf", 10.);
  desc.addUntracked<double>("minVertexMeanWeight", 0.5);
  desc.addUntracked<std::vector<double>>("profilePtBorders", {3, 5, 10, 20, 50, 100});
  desc.addUntracked<std::vector<int>>("iovList", {0, 500000});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetHTAnalyzer);
