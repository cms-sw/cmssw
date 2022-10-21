// -*- C++ -*-
//
// Package:    IsolatedParticles
// Class:      StudyCaloResponse
//
/**\class StudyCaloResponse StudyCaloResponse.cc Calibration/IsolatedParticles/plugins/StudyCaloResponse.cc

 Description: Studies single particle response measurements in data/MC

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Mar  4 18:52:02 CST 2011
//
//

// system include files
#include <memory>
#include <string>

// Root objects
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

// user include files
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

class StudyCaloResponse : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit StudyCaloResponse(const edm::ParameterSet&);
  ~StudyCaloResponse() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

  void clear();
  void fillTrack(int, double, double, double, double);
  void fillIsolation(int, double, double, double);
  void fillEnergy(int, int, double, double, double, double, double);
  std::string truncate_str(const std::string&);
  int trackPID(const reco::Track*, const edm::Handle<reco::GenParticleCollection>&);

  // ----------member data ---------------------------
  static const int nPBin_ = 15, nEtaBin_ = 4, nPVBin_ = 4;
  static const int nGen_ = (nPVBin_ + 12);
  HLTConfigProvider hltConfig_;
  edm::Service<TFileService> fs_;
  const int verbosity_;
  const std::vector<std::string> trigNames_, newNames_;
  const edm::InputTag labelMuon_, labelGenTrack_;
  const std::string theTrackQuality_;
  const double minTrackP_, maxTrackEta_;
  const double tMinE_, tMaxE_, tMinH_, tMaxH_;
  const double eThrEB_, eThrEE_, eThrHB_, eThrHE_;
  const bool isItAOD_, vetoTrigger_, doTree_, vetoMuon_, vetoEcal_;
  const double cutMuon_, cutEcal_, cutRatio_;
  const std::vector<double> puWeights_;
  const edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  spr::trackSelectionParameters selectionParameters_;
  std::vector<std::string> HLTNames_;
  bool changed_, firstEvent_;

  edm::EDGetTokenT<LumiDetails> tok_lumi;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct> tok_ew_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_ecalChStatus_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;

  TH1I *h_nHLT, *h_HLTAccept, *h_HLTCorr, *h_numberPV;
  TH1I *h_goodPV, *h_goodRun;
  TH2I* h_nHLTvsRN;
  std::vector<TH1I*> h_HLTAccepts;
  TH1D *h_p[nGen_ + 2], *h_pt[nGen_ + 2], *h_counter[8];
  TH1D *h_eta[nGen_ + 2], *h_phi[nGen_ + 2], *h_h_pNew[8];
  TH1I* h_ntrk[2];
  TH1D *h_maxNearP[2], *h_ene1[2], *h_ene2[2], *h_ediff[2];
  TH1D* h_energy[nPVBin_ + 8][nPBin_][nEtaBin_][6];
  TTree* tree_;
  int nRun_, etaBin_[nEtaBin_ + 1], pvBin_[nPVBin_ + 1];
  double pBin_[nPBin_ + 1];
  int tr_goodPV, tr_goodRun;
  double tr_eventWeight;
  std::vector<std::string> tr_TrigName;
  std::vector<double> tr_TrkPt, tr_TrkP, tr_TrkEta, tr_TrkPhi;
  std::vector<double> tr_MaxNearP31X31, tr_MaxNearHcalP7x7;
  std::vector<double> tr_H3x3, tr_H5x5, tr_H7x7;
  std::vector<double> tr_FE7x7P, tr_FE11x11P, tr_FE15x15P;
  std::vector<bool> tr_SE7x7P, tr_SE11x11P, tr_SE15x15P;
  std::vector<int> tr_iEta, tr_TrkID;
};

StudyCaloResponse::StudyCaloResponse(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getUntrackedParameter<int>("verbosity")),
      trigNames_(iConfig.getUntrackedParameter<std::vector<std::string> >("triggers")),
      newNames_(iConfig.getUntrackedParameter<std::vector<std::string> >("newNames")),
      labelMuon_(iConfig.getUntrackedParameter<edm::InputTag>("labelMuon")),
      labelGenTrack_(iConfig.getUntrackedParameter<edm::InputTag>("labelTrack")),
      theTrackQuality_(iConfig.getUntrackedParameter<std::string>("trackQuality")),
      minTrackP_(iConfig.getUntrackedParameter<double>("minTrackP")),
      maxTrackEta_(iConfig.getUntrackedParameter<double>("maxTrackEta")),
      tMinE_(iConfig.getUntrackedParameter<double>("timeMinCutECAL")),
      tMaxE_(iConfig.getUntrackedParameter<double>("timeMaxCutECAL")),
      tMinH_(iConfig.getUntrackedParameter<double>("timeMinCutHCAL")),
      tMaxH_(iConfig.getUntrackedParameter<double>("timeMaxCutHCAL")),
      eThrEB_(iConfig.getUntrackedParameter<double>("thresholdEB")),
      eThrEE_(iConfig.getUntrackedParameter<double>("thresholdEE")),
      eThrHB_(iConfig.getUntrackedParameter<double>("thresholdHB")),
      eThrHE_(iConfig.getUntrackedParameter<double>("thresholdHE")),
      isItAOD_(iConfig.getUntrackedParameter<bool>("isItAOD")),
      vetoTrigger_(iConfig.getUntrackedParameter<bool>("vetoTrigger")),
      doTree_(iConfig.getUntrackedParameter<bool>("doTree")),
      vetoMuon_(iConfig.getUntrackedParameter<bool>("vetoMuon")),
      vetoEcal_(iConfig.getUntrackedParameter<bool>("vetoEcal")),
      cutMuon_(iConfig.getUntrackedParameter<double>("cutMuon")),
      cutEcal_(iConfig.getUntrackedParameter<double>("cutEcal")),
      cutRatio_(iConfig.getUntrackedParameter<double>("cutRatio")),
      puWeights_(iConfig.getUntrackedParameter<std::vector<double> >("puWeights")),
      triggerEvent_(edm::InputTag("hltTriggerSummaryAOD", "", "HLT")),
      theTriggerResultsLabel_(edm::InputTag("TriggerResults", "", "HLT")),
      nRun_(0) {
  usesResource(TFileService::kSharedResource);

  reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameters_.minPt = iConfig.getUntrackedParameter<double>("minTrackPt");
  selectionParameters_.minQuality = trackQuality;
  selectionParameters_.maxDxyPV = iConfig.getUntrackedParameter<double>("maxDxyPV");
  selectionParameters_.maxDzPV = iConfig.getUntrackedParameter<double>("maxDzPV");
  selectionParameters_.maxChi2 = iConfig.getUntrackedParameter<double>("maxChi2");
  selectionParameters_.maxDpOverP = iConfig.getUntrackedParameter<double>("maxDpOverP");
  selectionParameters_.minOuterHit = iConfig.getUntrackedParameter<int>("minOuterHit");
  selectionParameters_.minLayerCrossed = iConfig.getUntrackedParameter<int>("minLayerCrossed");
  selectionParameters_.maxInMiss = iConfig.getUntrackedParameter<int>("maxInMiss");
  selectionParameters_.maxOutMiss = iConfig.getUntrackedParameter<int>("maxOutMiss");

  // define tokens for access
  tok_lumi = consumes<LumiDetails, edm::InLumi>(edm::InputTag("lumiProducer"));
  tok_trigEvt = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);
  tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_parts_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("particleSource"));

  if (isItAOD_) {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("reducedHcalRecHits", "hbhereco"));
  } else {
    tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
    tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
    tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  }
  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_topo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_ecalChStatus_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  tok_sevlv_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();

  edm::LogVerbatim("IsoTrack") << "Verbosity " << verbosity_ << " with " << trigNames_.size() << " triggers:";
  for (unsigned int k = 0; k < trigNames_.size(); ++k)
    edm::LogVerbatim("IsoTrack") << " [" << k << "] " << trigNames_[k];
  edm::LogVerbatim("IsoTrack") << "TrackQuality " << theTrackQuality_ << " Minpt " << selectionParameters_.minPt
                               << " maxDxy " << selectionParameters_.maxDxyPV << " maxDz "
                               << selectionParameters_.maxDzPV << " maxChi2 " << selectionParameters_.maxChi2
                               << " maxDp/p " << selectionParameters_.maxDpOverP << " minOuterHit "
                               << selectionParameters_.minOuterHit << " minLayerCrossed "
                               << selectionParameters_.minLayerCrossed << " maxInMiss "
                               << selectionParameters_.maxInMiss << " maxOutMiss " << selectionParameters_.maxOutMiss
                               << " minTrackP " << minTrackP_ << " maxTrackEta " << maxTrackEta_ << " tMinE_ " << tMinE_
                               << " tMaxE " << tMaxE_ << " tMinH_ " << tMinH_ << " tMaxH_ " << tMaxH_ << " isItAOD "
                               << isItAOD_ << " doTree " << doTree_ << " vetoTrigger " << vetoTrigger_ << " vetoMuon "
                               << vetoMuon_ << ":" << cutMuon_ << " vetoEcal " << vetoEcal_ << ":" << cutEcal_ << ":"
                               << cutRatio_;

  double pBins[nPBin_ + 1] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 100.0};
  int etaBins[nEtaBin_ + 1] = {1, 7, 13, 17, 23};
  int pvBins[nPVBin_ + 1] = {1, 2, 3, 5, 100};
  for (int i = 0; i <= nPBin_; ++i)
    pBin_[i] = pBins[i];
  for (int i = 0; i <= nEtaBin_; ++i)
    etaBin_[i] = etaBins[i];
  for (int i = 0; i <= nPVBin_; ++i)
    pvBin_[i] = pvBins[i];

  firstEvent_ = true;
  changed_ = false;
}

void StudyCaloResponse::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<std::string> trig;
  std::vector<double> weights;
  std::vector<std::string> newNames = {"HLT", "PixelTracks_Multiplicity", "HLT_Physics_", "HLT_JetE", "HLT_ZeroBias"};
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("particleSource", edm::InputTag("genParticles"));
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<std::vector<std::string> >("triggers", trig);
  desc.addUntracked<std::vector<std::string> >("newNames", newNames);
  desc.addUntracked<edm::InputTag>("labelMuon", edm::InputTag("muons", "", "RECO"));
  desc.addUntracked<edm::InputTag>("labelTrack", edm::InputTag("generalTracks", "", "RECO"));
  desc.addUntracked<std::string>("trackQuality", "highPurity");
  desc.addUntracked<double>("minTrackPt", 1.0);
  desc.addUntracked<double>("maxDxyPV", 0.02);
  desc.addUntracked<double>("maxDzPV", 0.02);
  desc.addUntracked<double>("maxChi2", 5.0);
  desc.addUntracked<double>("maxDpOverP", 0.1);
  desc.addUntracked<int>("minOuterHit", 4);
  desc.addUntracked<int>("minLayerCrossed", 8);
  desc.addUntracked<int>("maxInMiss", 0);
  desc.addUntracked<int>("maxOutMiss", 0);
  desc.addUntracked<double>("minTrackP", 1.0);
  desc.addUntracked<double>("maxTrackEta", 2.6);
  desc.addUntracked<double>("timeMinCutECAL", -500.0);
  desc.addUntracked<double>("timeMaxCutECAL", 500.0);
  desc.addUntracked<double>("timeMinCutHCAL", -500.0);
  desc.addUntracked<double>("timeMaxCutHCAL", 500.0);
  desc.addUntracked<double>("thresholdEB", 0.030);
  desc.addUntracked<double>("thresholdEE", 0.150);
  desc.addUntracked<double>("thresholdHB", 0.7);
  desc.addUntracked<double>("thresholdHE", 0.8);
  desc.addUntracked<bool>("isItAOD", false);
  desc.addUntracked<bool>("vetoTrigger", false);
  desc.addUntracked<bool>("doTree", false);
  desc.addUntracked<bool>("vetoMuon", true);
  desc.addUntracked<double>("cutMuon", 0.1);
  desc.addUntracked<bool>("vetoEcal", false);
  desc.addUntracked<double>("cutEcal", 2.0);
  desc.addUntracked<double>("cutRatio", 0.9);
  desc.addUntracked<std::vector<double> >("puWeights", weights);
  descriptions.add("studyCaloResponse", desc);
}

void StudyCaloResponse::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  clear();
  int counter0[1000] = {0};
  int counter1[1000] = {0};
  int counter2[1000] = {0};
  int counter3[1000] = {0};
  int counter4[1000] = {0};
  int counter5[1000] = {0};
  int counter6[1000] = {0};
  int counter7[1000] = {0};
  if (verbosity_ > 0)
    edm::LogVerbatim("IsoTrack") << "Event starts====================================";
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();
  int Lumi = iEvent.luminosityBlock();
  int Bunch = iEvent.bunchCrossing();

  std::vector<int> newAccept(newNames_.size() + 1, 0);
  if (verbosity_ > 0)
    edm::LogVerbatim("IsoTrack") << "RunNo " << RunNo << " EvtNo " << EvtNo << " Lumi " << Lumi << " Bunch " << Bunch;

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);

  bool ok(false);
  std::string triggerUse("None");
  if (!triggerEventHandle.isValid()) {
    if (trigNames_.empty()) {
      ok = true;
    } else {
      edm::LogWarning("StudyCaloResponse") << "Error! Can't get the product " << triggerEvent_.label();
    }
  } else {
    triggerEvent = *(triggerEventHandle.product());

    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);

    if (triggerResults.isValid()) {
      h_nHLT->Fill(triggerResults->size());
      h_nHLTvsRN->Fill(RunNo, triggerResults->size());

      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
        int ipos = -1;
        std::string newtriggerName = truncate_str(triggerNames_[iHLT]);
        for (unsigned int i = 0; i < HLTNames_.size(); ++i) {
          if (newtriggerName == HLTNames_[i]) {
            ipos = i + 1;
            break;
          }
        }
        if (ipos < 0) {
          HLTNames_.push_back(newtriggerName);
          ipos = (int)(HLTNames_.size());
          if (ipos <= h_HLTAccept->GetNbinsX())
            h_HLTAccept->GetXaxis()->SetBinLabel(ipos, newtriggerName.c_str());
        }
        if ((int)(iHLT + 1) > h_HLTAccepts[nRun_]->GetNbinsX()) {
          edm::LogVerbatim("IsoTrack") << "Wrong trigger " << RunNo << " Event " << EvtNo << " Hlt " << iHLT;
        } else {
          if (firstEvent_)
            h_HLTAccepts[nRun_]->GetXaxis()->SetBinLabel(iHLT + 1, newtriggerName.c_str());
        }
        int hlt = triggerResults->accept(iHLT);
        if (hlt) {
          h_HLTAccepts[nRun_]->Fill(iHLT + 1);
          h_HLTAccept->Fill(ipos);
        }
        if (trigNames_.empty()) {
          ok = true;
        } else {
          for (unsigned int i = 0; i < trigNames_.size(); ++i) {
            if (newtriggerName.find(trigNames_[i]) != std::string::npos) {
              if (verbosity_ % 10 > 0)
                edm::LogVerbatim("IsoTrack") << newtriggerName;
              if (hlt > 0) {
                if (!ok)
                  triggerUse = newtriggerName;
                ok = true;
                tr_TrigName.push_back(newtriggerName);
              }
            }
          }
          if (vetoTrigger_)
            ok = !ok;
          for (unsigned int i = 0; i < newNames_.size(); ++i) {
            if (newtriggerName.find(newNames_[i]) != std::string::npos) {
              if (verbosity_ % 10 > 0)
                edm::LogVerbatim("IsoTrack") << "[" << i << "] " << newNames_[i] << " : " << newtriggerName;
              if (hlt > 0)
                newAccept[i] = 1;
            }
          }
        }
      }
      int iflg(0), indx(1);
      for (unsigned int i = 0; i < newNames_.size(); ++i) {
        iflg += (indx * newAccept[i]);
        indx *= 2;
      }
      h_HLTCorr->Fill(iflg);
    }
  }
  if ((verbosity_ / 10) % 10 > 0)
    edm::LogVerbatim("IsoTrack") << "Trigger check gives " << ok << " with " << triggerUse;

  //Look at the tracks
  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  edm::Handle<reco::MuonCollection> muonEventHandle;
  iEvent.getByToken(tok_Muon_, muonEventHandle);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);

  if ((!trkCollection.isValid()) || (!muonEventHandle.isValid()) || (!recVtxs.isValid())) {
    edm::LogWarning("StudyCaloResponse") << "Track collection " << trkCollection.isValid() << " Muon collection "
                                         << muonEventHandle.isValid() << " Vertex Collecttion " << recVtxs.isValid();
    ok = false;
  }

  if (ok) {
    h_goodRun->Fill(RunNo);
    tr_goodRun = RunNo;
    // get handles to calogeometry and calotopology
    const CaloGeometry* geo = &iSetup.getData(tok_geom_);
    const CaloTopology* caloTopology = &iSetup.getData(tok_caloTopology_);
    const HcalTopology* theHBHETopology = &iSetup.getData(tok_topo_);
    const MagneticField* bField = &iSetup.getData(tok_magField_);
    const EcalChannelStatus* theEcalChStatus = &iSetup.getData(tok_ecalChStatus_);

    int ntrk(0), ngoodPV(0), nPV(-1), nvtxs(0);
    nvtxs = (int)(recVtxs->size());
    for (int ind = 0; ind < nvtxs; ind++) {
      if (!((*recVtxs)[ind].isFake()) && (*recVtxs)[ind].ndof() > 4)
        ngoodPV++;
    }
    for (int i = 0; i < nPVBin_; ++i) {
      if (ngoodPV >= pvBin_[i] && ngoodPV < pvBin_[i + 1]) {
        nPV = i;
        break;
      }
    }

    tr_eventWeight = 1.0;
    edm::Handle<GenEventInfoProduct> genEventInfo;
    iEvent.getByToken(tok_ew_, genEventInfo);
    if (genEventInfo.isValid())
      tr_eventWeight = genEventInfo->weight();

    if ((verbosity_ / 10) % 10 > 0)
      edm::LogVerbatim("IsoTrack") << "Number of vertices: " << nvtxs << " Good " << ngoodPV << " Bin " << nPV
                                   << " Event weight " << tr_eventWeight;
    h_numberPV->Fill(nvtxs, tr_eventWeight);
    h_goodPV->Fill(ngoodPV, tr_eventWeight);
    tr_goodPV = ngoodPV;

    if (!puWeights_.empty()) {
      int npbin = h_goodPV->FindBin(ngoodPV);
      if (npbin > 0 && npbin <= (int)(puWeights_.size()))
        tr_eventWeight *= puWeights_[npbin - 1];
      else
        tr_eventWeight = 0;
    }

    //=== genParticle information
    edm::Handle<reco::GenParticleCollection> genParticles;
    iEvent.getByToken(tok_parts_, genParticles);
    if (genParticles.isValid()) {
      for (const auto& p : (reco::GenParticleCollection)(*genParticles)) {
        double pt1 = p.momentum().Rho();
        double p1 = p.momentum().R();
        double eta1 = p.momentum().Eta();
        double phi1 = p.momentum().Phi();
        fillTrack(nGen_, pt1, p1, eta1, phi1);
        bool match(false);
        double phi2(phi1);
        if (phi2 < 0)
          phi2 += 2.0 * M_PI;
        for (const auto& trk : (reco::TrackCollection)(*trkCollection)) {
          bool quality = trk.quality(selectionParameters_.minQuality);
          if (quality) {
            double dEta = trk.eta() - eta1;
            double phi0 = trk.phi();
            if (phi0 < 0)
              phi0 += 2.0 * M_PI;
            double dPhi = phi0 - phi2;
            if (dPhi > M_PI)
              dPhi -= 2. * M_PI;
            else if (dPhi < -M_PI)
              dPhi += 2. * M_PI;
            double dR = sqrt(dEta * dEta + dPhi * dPhi);
            if (dR < 0.01) {
              match = true;
              break;
            }
          }
        }
        if (match)
          fillTrack(nGen_ + 1, pt1, p1, eta1, phi1);
      }
    }

    reco::TrackCollection::const_iterator trkItr;
    for (trkItr = trkCollection->begin(); trkItr != trkCollection->end(); ++trkItr, ++ntrk) {
      const reco::Track* pTrack = &(*trkItr);
      double pt1 = pTrack->pt();
      double p1 = pTrack->p();
      double eta1 = pTrack->momentum().eta();
      double phi1 = pTrack->momentum().phi();
      bool quality = pTrack->quality(selectionParameters_.minQuality);
      fillTrack(0, pt1, p1, eta1, phi1);
      if (quality)
        fillTrack(1, pt1, p1, eta1, phi1);
      if (p1 < 1000) {
        h_h_pNew[0]->Fill(p1);
        ++counter0[(int)(p1)];
      }
    }
    h_ntrk[0]->Fill(ntrk, tr_eventWeight);

    std::vector<spr::propagatedTrackID> trkCaloDets;
    spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDets, ((verbosity_ / 100) % 10 > 0));
    std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;
    for (trkDetItr = trkCaloDets.begin(), ntrk = 0; trkDetItr != trkCaloDets.end(); trkDetItr++, ntrk++) {
      const reco::Track* pTrack = &(*(trkDetItr->trkItr));
      double pt1 = pTrack->pt();
      double p1 = pTrack->p();
      double eta1 = pTrack->momentum().eta();
      double phi1 = pTrack->momentum().phi();
      if ((verbosity_ / 10) % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "track: p " << p1 << " pt " << pt1 << " eta " << eta1 << " phi " << phi1
                                     << " okEcal " << trkDetItr->okECAL;
      fillTrack(2, pt1, p1, eta1, phi1);

      bool vetoMuon(false);
      double chiGlobal(0), dr(0);
      bool goodGlob(false);
      if (vetoMuon_) {
        if (muonEventHandle.isValid()) {
          for (reco::MuonCollection::const_iterator recMuon = muonEventHandle->begin();
               recMuon != muonEventHandle->end();
               ++recMuon) {
            if (((recMuon->isPFMuon()) && (recMuon->isGlobalMuon() || recMuon->isTrackerMuon())) &&
                (recMuon->innerTrack()->validFraction() > 0.49) && (recMuon->innerTrack().isNonnull())) {
              chiGlobal = ((recMuon->globalTrack().isNonnull()) ? recMuon->globalTrack()->normalizedChi2() : 999);
              goodGlob = (recMuon->isGlobalMuon() && chiGlobal < 3 &&
                          recMuon->combinedQuality().chi2LocalPosition < 12 && recMuon->combinedQuality().trkKink < 20);
              if (muon::segmentCompatibility(*recMuon) > (goodGlob ? 0.303 : 0.451)) {
                const reco::Track* pTrack0 = (recMuon->innerTrack()).get();
                dr = reco::deltaR(pTrack0->eta(), pTrack0->phi(), pTrack->eta(), pTrack->phi());
                if (dr < cutMuon_) {
                  vetoMuon = true;
                  break;
                }
              }
            }
          }
        }
      }
      if ((verbosity_ / 10) % 10 > 0)
        edm::LogVerbatim("IsoTrack") << "vetoMuon: " << vetoMuon_ << ":" << vetoMuon << " chi:good:dr " << chiGlobal
                                     << ":" << goodGlob << ":" << dr;
      if (pt1 > minTrackP_ && std::abs(eta1) < maxTrackEta_ && trkDetItr->okECAL && (!vetoMuon)) {
        fillTrack(3, pt1, p1, eta1, phi1);
        double maxNearP31x31 =
            spr::chargeIsolationEcal(ntrk, trkCaloDets, geo, caloTopology, 15, 15, ((verbosity_ / 1000) % 10 > 0));

        const EcalSeverityLevelAlgo* sevlv = &iSetup.getData(tok_sevlv_);

        edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
        edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
        iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
        iEvent.getByToken(tok_EE_, endcapRecHitsHandle);
        // get ECal Tranverse Profile
        std::pair<double, bool> e7x7P, e11x11P, e15x15P;
        const DetId isoCell = trkDetItr->detIdECAL;
        e7x7P = spr::eECALmatrix(isoCell,
                                 barrelRecHitsHandle,
                                 endcapRecHitsHandle,
                                 *theEcalChStatus,
                                 geo,
                                 caloTopology,
                                 sevlv,
                                 3,
                                 3,
                                 eThrEB_,
                                 eThrEE_,
                                 tMinE_,
                                 tMaxE_,
                                 ((verbosity_ / 10000) % 10 > 0));
        e11x11P = spr::eECALmatrix(isoCell,
                                   barrelRecHitsHandle,
                                   endcapRecHitsHandle,
                                   *theEcalChStatus,
                                   geo,
                                   caloTopology,
                                   sevlv,
                                   5,
                                   5,
                                   eThrEB_,
                                   eThrEE_,
                                   tMinE_,
                                   tMaxE_,
                                   ((verbosity_ / 10000) % 10 > 0));
        e15x15P = spr::eECALmatrix(isoCell,
                                   barrelRecHitsHandle,
                                   endcapRecHitsHandle,
                                   *theEcalChStatus,
                                   geo,
                                   caloTopology,
                                   sevlv,
                                   7,
                                   7,
                                   eThrEB_,
                                   eThrEE_,
                                   tMinE_,
                                   tMaxE_,
                                   ((verbosity_ / 10000) % 10 > 0));

        double maxNearHcalP7x7 =
            spr::chargeIsolationHcal(ntrk, trkCaloDets, theHBHETopology, 3, 3, ((verbosity_ / 1000) % 10 > 0));
        int ieta(0);
        double h3x3(0), h5x5(0), h7x7(0);
        fillIsolation(0, maxNearP31x31, e11x11P.first, e15x15P.first);
        if ((verbosity_ / 10) % 10 > 0)
          edm::LogVerbatim("IsoTrack") << "Accepted Tracks reaching Ecal maxNearP31x31 " << maxNearP31x31 << " e11x11P "
                                       << e11x11P.first << " e15x15P " << e15x15P.first << " okHCAL "
                                       << trkDetItr->okHCAL;

        int trackID = trackPID(pTrack, genParticles);
        if (trkDetItr->okHCAL) {
          edm::Handle<HBHERecHitCollection> hbhe;
          iEvent.getByToken(tok_hbhe_, hbhe);
          const DetId ClosestCell(trkDetItr->detIdHCAL);
          ieta = ((HcalDetId)(ClosestCell)).ietaAbs();
          h3x3 = spr::eHCALmatrix(theHBHETopology,
                                  ClosestCell,
                                  hbhe,
                                  1,
                                  1,
                                  false,
                                  true,
                                  eThrHB_,
                                  eThrHE_,
                                  -100.0,
                                  -100.0,
                                  tMinH_,
                                  tMaxH_,
                                  ((verbosity_ / 10000) % 10 > 0));
          h5x5 = spr::eHCALmatrix(theHBHETopology,
                                  ClosestCell,
                                  hbhe,
                                  2,
                                  2,
                                  false,
                                  true,
                                  eThrHB_,
                                  eThrHE_,
                                  -100.0,
                                  -100.0,
                                  tMinH_,
                                  tMaxH_,
                                  ((verbosity_ / 10000) % 10 > 0));
          h7x7 = spr::eHCALmatrix(theHBHETopology,
                                  ClosestCell,
                                  hbhe,
                                  3,
                                  3,
                                  false,
                                  true,
                                  eThrHB_,
                                  eThrHE_,
                                  -100.0,
                                  -100.0,
                                  tMinH_,
                                  tMaxH_,
                                  ((verbosity_ / 10000) % 10 > 0));
          fillIsolation(1, maxNearHcalP7x7, h5x5, h7x7);
          double eByh = ((e11x11P.second) ? (e11x11P.first / std::max(h3x3, 0.001)) : 0.0);
          bool notAnElec = ((vetoEcal_ && e11x11P.second) ? ((e11x11P.first < cutEcal_) || (eByh < cutRatio_)) : true);
          if ((verbosity_ / 10) % 10 > 0)
            edm::LogVerbatim("IsoTrack") << "Tracks Reaching Hcal maxNearHcalP7x7/h5x5/h7x7 " << maxNearHcalP7x7 << "/"
                                         << h5x5 << "/" << h7x7 << " eByh " << eByh << " notAnElec " << notAnElec;
          tr_TrkPt.push_back(pt1);
          tr_TrkP.push_back(p1);
          tr_TrkEta.push_back(eta1);
          tr_TrkPhi.push_back(phi1);
          tr_TrkID.push_back(trackID);
          tr_MaxNearP31X31.push_back(maxNearP31x31);
          tr_MaxNearHcalP7x7.push_back(maxNearHcalP7x7);
          tr_FE7x7P.push_back(e7x7P.first);
          tr_FE11x11P.push_back(e11x11P.first);
          tr_FE15x15P.push_back(e15x15P.first);
          tr_SE7x7P.push_back(e7x7P.second);
          tr_SE11x11P.push_back(e11x11P.second);
          tr_SE15x15P.push_back(e15x15P.second);
          tr_iEta.push_back(ieta);
          tr_H3x3.push_back(h3x3);
          tr_H5x5.push_back(h5x5);
          tr_H7x7.push_back(h7x7);

          if (maxNearP31x31 < 0 && notAnElec) {
            fillTrack(4, pt1, p1, eta1, phi1);
            fillEnergy(0, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
            if (maxNearHcalP7x7 < 0) {
              fillTrack(5, pt1, p1, eta1, phi1);
              fillEnergy(1, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
              if ((e11x11P.second) && (e15x15P.second) && (e15x15P.first - e11x11P.first) < 2.0) {
                fillTrack(6, pt1, p1, eta1, phi1);
                fillEnergy(2, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
                if (h7x7 - h5x5 < 2.0) {
                  fillTrack(7, pt1, p1, eta1, phi1);
                  fillEnergy(3, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
                  if (nPV >= 0) {
                    fillTrack(nPV + 8, pt1, p1, eta1, phi1);
                    fillEnergy(nPV + 4, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
                  }
                  if (trackID > 0) {
                    fillTrack(nPVBin_ + trackID + 7, pt1, p1, eta1, phi1);
                    fillEnergy(nPVBin_ + trackID + 3, ieta, p1, e7x7P.first, h3x3, e11x11P.first, h5x5);
                  }
                  if (p1 < 1000) {
                    h_h_pNew[7]->Fill(p1);
                    ++counter7[(int)(p1)];
                  }
                }
                if (p1 < 1000) {
                  h_h_pNew[6]->Fill(p1);
                  ++counter6[(int)(p1)];
                }
              }
              if (p1 < 1000) {
                h_h_pNew[5]->Fill(p1);
                ++counter5[(int)(p1)];
              }
            }
            if (p1 < 1000) {
              h_h_pNew[4]->Fill(p1);
              ++counter4[(int)(p1)];
            }
          }
          if (p1 < 1000) {
            h_h_pNew[3]->Fill(p1);
            ++counter3[(int)(p1)];
          }
        }
        if (p1 < 1000) {
          h_h_pNew[2]->Fill(p1);
          ++counter2[(int)(p1)];
        }
      }
      if (p1 < 1000) {
        h_h_pNew[1]->Fill(p1);
        ++counter1[(int)(p1)];
      }
    }
    h_ntrk[1]->Fill(ntrk, tr_eventWeight);
    if ((!tr_TrkPt.empty()) && doTree_)
      tree_->Fill();
    for (int i = 0; i < 1000; ++i) {
      if (counter0[i])
        h_counter[0]->Fill(i, counter0[i]);
      if (counter1[i])
        h_counter[1]->Fill(i, counter1[i]);
      if (counter2[i])
        h_counter[2]->Fill(i, counter2[i]);
      if (counter3[i])
        h_counter[3]->Fill(i, counter3[i]);
      if (counter4[i])
        h_counter[4]->Fill(i, counter4[i]);
      if (counter5[i])
        h_counter[5]->Fill(i, counter5[i]);
      if (counter6[i])
        h_counter[6]->Fill(i, counter6[i]);
      if (counter7[i])
        h_counter[7]->Fill(i, counter7[i]);
    }
  }
  firstEvent_ = false;
}

void StudyCaloResponse::beginJob() {
  // Book histograms
  h_nHLT = fs_->make<TH1I>("h_nHLT", "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept = fs_->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    h_HLTAccept->GetXaxis()->SetBinLabel(i, " ");
  h_nHLTvsRN = fs_->make<TH2I>("h_nHLTvsRN", "size of trigger Names vs RunNo", 2168, 190949, 193116, 100, 400, 500);
  h_HLTCorr = fs_->make<TH1I>("h_HLTCorr", "Correlation among different paths", 100, 0, 100);
  h_numberPV = fs_->make<TH1I>("h_numberPV", "Number of Primary Vertex", 100, 0, 100);
  h_goodPV = fs_->make<TH1I>("h_goodPV", "Number of good Primary Vertex", 100, 0, 100);
  h_goodRun = fs_->make<TH1I>("h_goodRun", "Number of accepted events for Run", 4000, 190000, 1940000);
  char hname[60], htit[200];
  std::string CollectionNames[2] = {"Reco", "Propagated"};
  for (unsigned int i = 0; i < 2; i++) {
    sprintf(hname, "h_nTrk_%s", CollectionNames[i].c_str());
    sprintf(htit, "Number of %s tracks", CollectionNames[i].c_str());
    h_ntrk[i] = fs_->make<TH1I>(hname, htit, 500, 0, 500);
  }
  std::string TrkNames[8] = {
      "All", "Quality", "NoIso", "okEcal", "EcalCharIso", "HcalCharIso", "EcalNeutIso", "HcalNeutIso"};
  std::string particle[4] = {"Electron", "Pion", "Kaon", "Proton"};
  for (unsigned int i = 0; i <= nGen_ + 1; i++) {
    if (i < 8) {
      sprintf(hname, "h_pt_%s", TrkNames[i].c_str());
      sprintf(htit, "p_{T} of %s tracks", TrkNames[i].c_str());
    } else if (i < 8 + nPVBin_) {
      sprintf(hname, "h_pt_%s_%d", TrkNames[7].c_str(), i - 8);
      sprintf(htit, "p_{T} of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin_[i - 8], pvBin_[i - 7] - 1);
    } else if (i >= nGen_) {
      sprintf(hname, "h_pt_%s_%d", TrkNames[0].c_str(), i - nGen_);
      sprintf(htit, "p_{T} of %s Generator tracks", TrkNames[0].c_str());
    } else {
      sprintf(hname, "h_pt_%s_%s", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
      sprintf(htit, "p_{T} of %s tracks (%s)", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
    }
    h_pt[i] = fs_->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_pt[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_p_%s", TrkNames[i].c_str());
      sprintf(htit, "Momentum of %s tracks", TrkNames[i].c_str());
    } else if (i < 8 + nPVBin_) {
      sprintf(hname, "h_p_%s_%d", TrkNames[7].c_str(), i - 8);
      sprintf(htit, "Momentum of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin_[i - 8], pvBin_[i - 7] - 1);
    } else if (i >= nGen_) {
      sprintf(hname, "h_p_%s_%d", TrkNames[0].c_str(), i - nGen_);
      sprintf(htit, "Momentum of %s Generator tracks", TrkNames[0].c_str());
    } else {
      sprintf(hname, "h_p_%s_%s", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
      sprintf(htit, "Momentum of %s tracks (%s)", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
    }
    h_p[i] = fs_->make<TH1D>(hname, htit, 400, 0, 200.0);
    h_p[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_eta_%s", TrkNames[i].c_str());
      sprintf(htit, "Eta of %s tracks", TrkNames[i].c_str());
    } else if (i < 8 + nPVBin_) {
      sprintf(hname, "h_eta_%s_%d", TrkNames[7].c_str(), i - 8);
      sprintf(htit, "Eta of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin_[i - 8], pvBin_[i - 7] - 1);
    } else if (i >= nGen_) {
      sprintf(hname, "h_eta_%s_%d", TrkNames[0].c_str(), i - nGen_);
      sprintf(htit, "Eta of %s Generator tracks", TrkNames[0].c_str());
    } else {
      sprintf(hname, "h_eta_%s_%s", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
      sprintf(htit, "Eta of %s tracks (%s)", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
    }
    h_eta[i] = fs_->make<TH1D>(hname, htit, 60, -3.0, 3.0);
    h_eta[i]->Sumw2();

    if (i < 8) {
      sprintf(hname, "h_phi_%s", TrkNames[i].c_str());
      sprintf(htit, "Phi of %s tracks", TrkNames[i].c_str());
    } else if (i < 8 + nPVBin_) {
      sprintf(hname, "h_phi_%s_%d", TrkNames[7].c_str(), i - 8);
      sprintf(htit, "Phi of %s tracks (PV=%d:%d)", TrkNames[7].c_str(), pvBin_[i - 8], pvBin_[i - 7] - 1);
    } else if (i >= nGen_) {
      sprintf(hname, "h_phi_%s_%d", TrkNames[0].c_str(), i - nGen_);
      sprintf(htit, "Phi of %s Generator tracks", TrkNames[0].c_str());
    } else {
      sprintf(hname, "h_phi_%s_%s", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
      sprintf(htit, "Phi of %s tracks (%s)", TrkNames[7].c_str(), particle[i - 8 - nPVBin_].c_str());
    }
    h_phi[i] = fs_->make<TH1D>(hname, htit, 100, -3.15, 3.15);
    h_phi[i]->Sumw2();
  }
  std::string IsolationNames[2] = {"Ecal", "Hcal"};
  for (unsigned int i = 0; i < 2; i++) {
    sprintf(hname, "h_maxNearP_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in ChargeIso region for %s", IsolationNames[i].c_str());
    h_maxNearP[i] = fs_->make<TH1D>(hname, htit, 120, -1.5, 10.5);
    h_maxNearP[i]->Sumw2();

    sprintf(hname, "h_ene1_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in smaller cone for %s", IsolationNames[i].c_str());
    h_ene1[i] = fs_->make<TH1D>(hname, htit, 400, 0.0, 200.0);
    h_ene1[i]->Sumw2();

    sprintf(hname, "h_ene2_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in bigger cone for %s", IsolationNames[i].c_str());
    h_ene2[i] = fs_->make<TH1D>(hname, htit, 400, 0.0, 200.0);
    h_ene2[i]->Sumw2();

    sprintf(hname, "h_ediff_%s", IsolationNames[i].c_str());
    sprintf(htit, "Energy in NeutralIso region for %s", IsolationNames[i].c_str());
    h_ediff[i] = fs_->make<TH1D>(hname, htit, 100, -0.5, 19.5);
    h_ediff[i]->Sumw2();
  }
  std::string energyNames[6] = {
      "E_{7x7}", "H_{3x3}", "(E_{7x7}+H_{3x3})", "E_{11x11}", "H_{5x5}", "{E_{11x11}+H_{5x5})"};
  for (int i = 0; i < 4 + nPVBin_ + 4; ++i) {
    for (int ip = 0; ip < nPBin_; ++ip) {
      for (int ie = 0; ie < nEtaBin_; ++ie) {
        for (int j = 0; j < 6; ++j) {
          sprintf(hname, "h_energy_%d_%d_%d_%d", i, ip, ie, j);
          if (i < 4) {
            sprintf(htit,
                    "%s/p (p=%4.1f:%4.1f; i#eta=%d:%d) for tracks with %s",
                    energyNames[j].c_str(),
                    pBin_[ip],
                    pBin_[ip + 1],
                    etaBin_[ie],
                    (etaBin_[ie + 1] - 1),
                    TrkNames[i + 4].c_str());
          } else if (i < 4 + nPVBin_) {
            sprintf(htit,
                    "%s/p (p=%4.1f:%4.1f, i#eta=%d:%d, PV=%d:%d) for tracks with %s",
                    energyNames[j].c_str(),
                    pBin_[ip],
                    pBin_[ip + 1],
                    etaBin_[ie],
                    (etaBin_[ie + 1] - 1),
                    pvBin_[i - 4],
                    (pvBin_[i - 3] - 1),
                    TrkNames[7].c_str());
          } else {
            sprintf(htit,
                    "%s/p (p=%4.1f:%4.1f, i#eta=%d:%d %s) for tracks with %s",
                    energyNames[j].c_str(),
                    pBin_[ip],
                    pBin_[ip + 1],
                    etaBin_[ie],
                    (etaBin_[ie + 1] - 1),
                    particle[i - 4 - nPVBin_].c_str(),
                    TrkNames[7].c_str());
          }
          h_energy[i][ip][ie][j] = fs_->make<TH1D>(hname, htit, 5000, -0.1, 49.9);
          h_energy[i][ip][ie][j]->Sumw2();
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    sprintf(hname, "counter%d", i);
    sprintf(htit, "Counter with cut %d", i);
    h_counter[i] = fs_->make<TH1D>(hname, htit, 1000, 0, 1000);
    sprintf(hname, "h_pTNew%d", i);
    sprintf(htit, "Track momentum with cut %d", i);
    h_h_pNew[i] = fs_->make<TH1D>(hname, htit, 1000, 0, 1000);
  }

  // Now the tree
  if (doTree_) {
    tree_ = fs_->make<TTree>("testTree", "new HLT Tree");
    tree_->Branch("tr_goodRun", &tr_goodRun, "tr_goodRun/I");
    tree_->Branch("tr_goodPV", &tr_goodPV, "tr_goodPV/I");
    tree_->Branch("tr_eventWeight", &tr_eventWeight, "tr_eventWeight/D");
    tree_->Branch("tr_tr_TrigName", &tr_TrigName);
    tree_->Branch("tr_TrkPt", &tr_TrkPt);
    tree_->Branch("tr_TrkP", &tr_TrkP);
    tree_->Branch("tr_TrkEta", &tr_TrkEta);
    tree_->Branch("tr_TrkPhi", &tr_TrkPhi);
    tree_->Branch("tr_TrkID", &tr_TrkID);
    tree_->Branch("tr_MaxNearP31X31", &tr_MaxNearP31X31);
    tree_->Branch("tr_MaxNearHcalP7x7", &tr_MaxNearHcalP7x7);
    tree_->Branch("tr_FE7x7P", &tr_FE7x7P);
    tree_->Branch("tr_FE11x11P", &tr_FE11x11P);
    tree_->Branch("tr_FE15x15P", &tr_FE15x15P);
    tree_->Branch("tr_SE7x7P", &tr_SE7x7P);
    tree_->Branch("tr_SE11x11P", &tr_SE11x11P);
    tree_->Branch("tr_SE15x15P", &tr_SE15x15P);
    tree_->Branch("tr_H3x3", &tr_H3x3);
    tree_->Branch("tr_H5x5", &tr_H5x5);
    tree_->Branch("tr_H7x7", &tr_H7x7);
    tree_->Branch("tr_iEta", &tr_iEta);
  }
}

// ------------ method called when starting to processes a run  ------------
void StudyCaloResponse::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  char hname[100], htit[400];
  edm::LogVerbatim("IsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                               << hltConfig_.init(iRun, iSetup, "HLT", changed_);
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I* hnew = fs_->make<TH1I>(hname, htit, 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    hnew->GetXaxis()->SetBinLabel(i, " ");
  h_HLTAccepts.push_back(hnew);
  edm::LogVerbatim("IsoTrack") << "beginRun " << iRun.run();
  firstEvent_ = true;
  changed_ = false;
}

// ------------ method called when ending the processing of a run  ------------
void StudyCaloResponse::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("IsoTrack") << "endRun[" << nRun_ << "] " << iRun.run();
}

void StudyCaloResponse::clear() {
  tr_TrigName.clear();
  tr_TrkPt.clear();
  tr_TrkP.clear();
  tr_TrkEta.clear();
  tr_TrkPhi.clear();
  tr_TrkID.clear();
  tr_MaxNearP31X31.clear();
  tr_MaxNearHcalP7x7.clear();
  tr_FE7x7P.clear();
  tr_FE11x11P.clear();
  tr_FE15x15P.clear();
  tr_SE7x7P.clear();
  tr_SE11x11P.clear();
  tr_SE15x15P.clear();
  tr_H3x3.clear();
  tr_H5x5.clear();
  tr_H7x7.clear();
  tr_iEta.clear();
}

void StudyCaloResponse::fillTrack(int i, double pt, double p, double eta, double phi) {
  h_pt[i]->Fill(pt, tr_eventWeight);
  h_p[i]->Fill(p, tr_eventWeight);
  h_eta[i]->Fill(eta, tr_eventWeight);
  h_phi[i]->Fill(phi, tr_eventWeight);
}

void StudyCaloResponse::fillIsolation(int i, double emaxnearP, double eneutIso1, double eneutIso2) {
  h_maxNearP[i]->Fill(emaxnearP, tr_eventWeight);
  h_ene1[i]->Fill(eneutIso1, tr_eventWeight);
  h_ene2[i]->Fill(eneutIso2, tr_eventWeight);
  h_ediff[i]->Fill(eneutIso2 - eneutIso1, tr_eventWeight);
}

void StudyCaloResponse::fillEnergy(
    int flag, int ieta, double p, double enEcal1, double enHcal1, double enEcal2, double enHcal2) {
  int ip(-1), ie(-1);
  for (int i = 0; i < nPBin_; ++i) {
    if (p >= pBin_[i] && p < pBin_[i + 1]) {
      ip = i;
      break;
    }
  }
  for (int i = 0; i < nEtaBin_; ++i) {
    if (ieta >= etaBin_[i] && ieta < etaBin_[i + 1]) {
      ie = i;
      break;
    }
  }
  if (ip >= 0 && ie >= 0 && enEcal1 > 0.02 && enHcal1 > 0.1) {
    h_energy[flag][ip][ie][0]->Fill(enEcal1 / p, tr_eventWeight);
    h_energy[flag][ip][ie][1]->Fill(enHcal1 / p, tr_eventWeight);
    h_energy[flag][ip][ie][2]->Fill((enEcal1 + enHcal1) / p, tr_eventWeight);
    h_energy[flag][ip][ie][3]->Fill(enEcal2 / p, tr_eventWeight);
    h_energy[flag][ip][ie][4]->Fill(enHcal2 / p, tr_eventWeight);
    h_energy[flag][ip][ie][5]->Fill((enEcal2 + enHcal2) / p, tr_eventWeight);
  }
}

std::string StudyCaloResponse::truncate_str(const std::string& str) {
  std::string truncated_str(str);
  int length = str.length();
  for (int i = 0; i < length - 2; i++) {
    if (str[i] == '_' && str[i + 1] == 'v' && isdigit(str.at(i + 2))) {
      int z = i + 1;
      truncated_str = str.substr(0, z);
    }
  }
  return (truncated_str);
}

int StudyCaloResponse::trackPID(const reco::Track* pTrack,
                                const edm::Handle<reco::GenParticleCollection>& genParticles) {
  int id(0);
  if (genParticles.isValid()) {
    unsigned int indx;
    reco::GenParticleCollection::const_iterator p;
    double mindR(999.9);
    for (p = genParticles->begin(), indx = 0; p != genParticles->end(); ++p, ++indx) {
      int pdgId = std::abs(p->pdgId());
      int idx = (pdgId == 11) ? 1 : ((pdgId == 211) ? 2 : ((pdgId == 321) ? 3 : ((pdgId == 2212) ? 4 : 0)));
      if (idx > 0) {
        double dEta = pTrack->eta() - p->momentum().Eta();
        double phi1 = pTrack->phi();
        double phi2 = p->momentum().Phi();
        if (phi1 < 0)
          phi1 += 2.0 * M_PI;
        if (phi2 < 0)
          phi2 += 2.0 * M_PI;
        double dPhi = phi1 - phi2;
        if (dPhi > M_PI)
          dPhi -= 2. * M_PI;
        else if (dPhi < -M_PI)
          dPhi += 2. * M_PI;
        double dR = sqrt(dEta * dEta + dPhi * dPhi);
        if (dR < mindR) {
          mindR = dR;
          id = idx;
        }
      }
    }
  }
  return id;
}

DEFINE_FWK_MODULE(StudyCaloResponse);
