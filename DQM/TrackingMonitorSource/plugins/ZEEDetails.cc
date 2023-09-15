// system includes
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// ROOT includes
#include "TFile.h"
#include "TLorentzVector.h"
#include "TMath.h"

class ZEEDetails : public DQMEDAnalyzer {
public:
  ZEEDetails(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  const std::string moduleName_;
  const std::string folderName_;

  const edm::InputTag electronTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag puSummaryTag_;
  const edm::InputTag vertexTag_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > puSummaryToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  const double maxEta_;
  const double minPt_;
  const double maxDeltaPhiInEB_;
  const double maxDeltaEtaInEB_;
  const double maxHOEEB_;
  const double maxSigmaiEiEEB_;
  const double maxDeltaPhiInEE_;
  const double maxDeltaEtaInEE_;
  const double maxHOEEE_;
  const double maxSigmaiEiEEE_;
  const double maxNormChi2_;
  const double maxD0_;
  const double maxDz_;
  const int minPixelHits_;
  const int minStripHits_;
  const double maxIso_;
  const double minPtHighest_;
  const double minInvMass_;
  const double maxInvMass_;
  const std::string trackQuality_;
  std::vector<float> vpu_;
  std::vector<float> vtrack_;
  const bool isMC_;
  const bool doPUCorrection_;
  const std::string puScaleFactorFile_;

  MonitorElement* Zpt_;
  MonitorElement* ZInvMass_;
  MonitorElement* EoverP_;
};

void ZEEDetails::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("moduleName", "ZEEDetails");
  desc.addUntracked<std::string>("folderName", "ElectronTracks");
  desc.addUntracked<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"));
  desc.addUntracked<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"));
  desc.addUntracked<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<double>("maxEta", 2.4);
  desc.addUntracked<double>("minPt", 5);
  desc.addUntracked<double>("maxDeltaPhiInEB", 0.15);
  desc.addUntracked<double>("maxDeltaEtaInEB", 0.007);
  desc.addUntracked<double>("maxHOEEB", 0.12);
  desc.addUntracked<double>("maxSigmaiEiEEB", 0.01);
  desc.addUntracked<double>("maxDeltaPhiInEE", 0.1);
  desc.addUntracked<double>("maxDeltaEtaInEE", 0.009);
  desc.addUntracked<double>("maxHOEEB_", .10);
  desc.addUntracked<double>("maxSigmaiEiEEE", 0.03);
  desc.addUntracked<double>("maxNormChi2", 10);
  desc.addUntracked<double>("maxD0", 0.02);
  desc.addUntracked<double>("maxDz", 20.);
  desc.addUntracked<uint32_t>("minPixelHits", 1);
  desc.addUntracked<uint32_t>("minStripHits", 8);
  desc.addUntracked<double>("maxIso", 0.3);
  desc.addUntracked<double>("minPtHighest", 24);
  desc.addUntracked<double>("minInvMass", 60);
  desc.addUntracked<double>("maxInvMass", 120);
  desc.addUntracked<std::string>("trackQuality", "highPurity");
  desc.addUntracked<bool>("isMC", false);
  desc.addUntracked<bool>("doPUCorrection", false);
  desc.addUntracked<std::string>("puScaleFactorFile", "PileupScaleFactor.root");
  descriptions.addWithDefaultLabel(desc);
}

ZEEDetails::ZEEDetails(const edm::ParameterSet& ps)
    : moduleName_(ps.getUntrackedParameter<std::string>("moduleName", "ZEEDetails")),
      folderName_(ps.getUntrackedParameter<std::string>("folderName", "ElectronTracks")),
      electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      puSummaryTag_(ps.getUntrackedParameter<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"))),
      vertexTag_(ps.getUntrackedParameter<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      puSummaryToken_(consumes<std::vector<PileupSummaryInfo> >(puSummaryTag_)),
      vertexToken_(consumes<reco::VertexCollection>(vertexTag_)),
      maxEta_(ps.getUntrackedParameter<double>("maxEta", 2.4)),
      minPt_(ps.getUntrackedParameter<double>("minPt", 5)),
      maxDeltaPhiInEB_(ps.getUntrackedParameter<double>("maxDeltaPhiInEB", 0.15)),
      maxDeltaEtaInEB_(ps.getUntrackedParameter<double>("maxDeltaEtaInEB", 0.007)),
      maxHOEEB_(ps.getUntrackedParameter<double>("maxHOEEB", 0.12)),
      maxSigmaiEiEEB_(ps.getUntrackedParameter<double>("maxSigmaiEiEEB", 0.01)),
      maxDeltaPhiInEE_(ps.getUntrackedParameter<double>("maxDeltaPhiInEE", 0.1)),
      maxDeltaEtaInEE_(ps.getUntrackedParameter<double>("maxDeltaEtaInEE", 0.009)),
      maxHOEEE_(ps.getUntrackedParameter<double>("maxHOEEB_", .10)),
      maxSigmaiEiEEE_(ps.getUntrackedParameter<double>("maxSigmaiEiEEE", 0.03)),
      maxNormChi2_(ps.getUntrackedParameter<double>("maxNormChi2", 10)),
      maxD0_(ps.getUntrackedParameter<double>("maxD0", 0.02)),
      maxDz_(ps.getUntrackedParameter<double>("maxDz", 20.)),
      minPixelHits_(ps.getUntrackedParameter<uint32_t>("minPixelHits", 1)),
      minStripHits_(ps.getUntrackedParameter<uint32_t>("minStripHits", 8)),
      maxIso_(ps.getUntrackedParameter<double>("maxIso", 0.3)),
      minPtHighest_(ps.getUntrackedParameter<double>("minPtHighest", 24)),
      minInvMass_(ps.getUntrackedParameter<double>("minInvMass", 60)),
      maxInvMass_(ps.getUntrackedParameter<double>("maxInvMass", 120)),
      trackQuality_(ps.getUntrackedParameter<std::string>("trackQuality", "highPurity")),
      isMC_(ps.getUntrackedParameter<bool>("isMC", false)),
      doPUCorrection_(ps.getUntrackedParameter<bool>("doPUCorrection", false)),
      puScaleFactorFile_(ps.getUntrackedParameter<std::string>("puScaleFactorFile", "PileupScaleFactor.root")) {
  if (isMC_ && doPUCorrection_) {
    vpu_.clear();
    TFile* f1 = TFile::Open(puScaleFactorFile_.c_str());
    TH1F* h1 = dynamic_cast<TH1F*>(f1->Get("pileupweight"));
    for (int i = 1; i <= h1->GetNbinsX(); ++i)
      vpu_.push_back(h1->GetBinContent(i));
    f1->Close();
  }
}

void ZEEDetails::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  std::string currentFolder = moduleName_ + "/" + folderName_;
  ibook.setCurrentFolder(currentFolder);
  Zpt_ = ibook.book1D("Zpt", "Z-Boson p_{T}", 100, 0.0, 100.0);
  ZInvMass_ = ibook.book1D("ZInvMass", "m_{ee}", 200, minInvMass_, maxInvMass_);
  EoverP_ = ibook.book3D("EoverP", "EoverP", 48, -2.4, 2.4, 36, -3.2, 3.2, 100, 0, 10);
}

void ZEEDetails::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  std::vector<TLorentzVector> list;
  std::vector<int> chrgeList;
  std::vector<reco::GsfElectron> finalelectrons;

  // Read Electron Collection
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  if (electronColl.isValid()) {
    for (auto const& ele : *electronColl) {
      if (!ele.ecalDriven())
        continue;
      if (ele.pt() < minPt_)
        continue;
      // set a max Eta cut
      if (!(ele.isEB() || ele.isEE()))
        continue;

      double hOverE = ele.hadronicOverEm();
      double sigmaee = ele.sigmaIetaIeta();
      double deltaPhiIn = ele.deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = ele.deltaEtaSuperClusterTrackAtVtx();

      // separate cut for barrel and endcap
      if (ele.isEB()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEB_ && fabs(deltaEtaIn) >= maxDeltaEtaInEB_ && hOverE >= maxHOEEB_ &&
            sigmaee >= maxSigmaiEiEEB_)
          continue;
      } else if (ele.isEE()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEE_ && fabs(deltaEtaIn) >= maxDeltaEtaInEE_ && hOverE >= maxHOEEE_ &&
            sigmaee >= maxSigmaiEiEEE_)
          continue;
      }

      reco::GsfTrackRef trk = ele.gsfTrack();
      reco::TrackRef tk = ele.closestCtfTrackRef();
      if (!trk.isNonnull())
        continue;  // only electrons with tracks
      if (!tk.isNonnull())
        continue;
      double chi2 = trk->chi2();
      double ndof = trk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      if (chbyndof >= maxNormChi2_)
        continue;

      double trkd0 = trk->d0();
      if (beamSpot.isValid()) {
        trkd0 = -(trk->dxy(beamSpot->position()));
      } else {
        edm::LogError("ElectronTrackProducer") << "Error >> Failed to get BeamSpot for label: " << bsTag_;
      }
      if (std::fabs(trkd0) >= maxD0_)
        continue;

      const reco::HitPattern& hitp = trk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      if (nPixelHits < minPixelHits_)
        continue;

      int nStripHits = hitp.numberOfValidStripHits();
      if (nStripHits < minStripHits_)
        continue;

      // DB corrected PF Isolation
      reco::GsfElectron::PflowIsolationVariables pfIso = ele.pfIsolationVariables();
      const float eiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      if (eiso > maxIso_ * ele.pt())
        continue;

      if (!tk->quality(reco::Track::qualityByName(trackQuality_)))
        continue;

      TLorentzVector lv;
      lv.SetPtEtaPhiE(ele.pt(), ele.eta(), ele.phi(), ele.energy());
      list.push_back(lv);
      chrgeList.push_back(ele.charge());
      finalelectrons.push_back(ele);
    }
  } else {
    edm::LogError("ElectronTrackProducer") << "Error >> Failed to get ElectronCollection for label: " << electronTag_;
  }

  edm::Handle<reco::VertexCollection> vertexColl;
  iEvent.getByToken(vertexToken_, vertexColl);
  if (!vertexColl.isValid()) {
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Vertex Collection, " << vertexTag_;
  }
  if (vertexColl->empty()) {
    edm::LogError("DqmTrackStudy") << "No good vertex in the event!!";
    return;
  }

  // Access PU information
  double wfac = 1.0;  // for data
  if (!iEvent.isRealData()) {
    edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
    iEvent.getByToken(puSummaryToken_, PupInfo);

    if (PupInfo.isValid()) {
      for (auto const& v : *PupInfo) {
        int bx = v.getBunchCrossing();
        if (bx == 0) {
          int nVertex = (vertexColl.isValid() ? vertexColl->size() : 0);
          if (doPUCorrection_) {
            if (nVertex > -1 && nVertex < int(vpu_.size()))
              wfac = vpu_.at(nVertex);
            else
              wfac = 0.0;
          }
        }
      }
    } else
      edm::LogError("DqmTrackStudy") << "PUSummary for input tag: " << puSummaryTag_ << " not found!!";
  }

  for (unsigned int I = 0; I != finalelectrons.size(); I++) {
    EoverP_->Fill(finalelectrons[I].superCluster()->eta(),
                  finalelectrons[I].superCluster()->phi(),
                  finalelectrons[I].eEleClusterOverPout(),
                  wfac);
  }

  if (list.size() >= 2) {
    if (chrgeList[0] + chrgeList[1] == 0) {
      if (list[0].Pt() >= minPtHighest_) {
        TLorentzVector zv = list[0] + list[1];
        if ((zv.M() >= minInvMass_) && (zv.M() <= maxInvMass_)) {
          Zpt_->Fill(zv.Pt(), wfac);
          ZInvMass_->Fill(zv.Mag(), wfac);
        }
      }
    }
  }
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ZEEDetails);
