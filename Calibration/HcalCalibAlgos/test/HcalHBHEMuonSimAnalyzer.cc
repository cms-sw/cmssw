#include <memory>
#include <iostream>
#include <vector>

#include <TTree.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#define EDM_ML_DEBUG

class HcalHBHEMuonSimAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalHBHEMuonSimAnalyzer(const edm::ParameterSet&);
  ~HcalHBHEMuonSimAnalyzer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  void clearVectors();
  unsigned int matchId(const HcalDetId&, const HcalDetId&);
  double activeLength(const DetId&);

  const std::string g4Label_, ebLabel_, eeLabel_, hcLabel_;
  const int verbosity_, maxDepth_;
  const double etaMax_, tMinE_, tMaxE_, tMinH_, tMaxH_;
  const edm::EDGetTokenT<edm::SimTrackContainer> tok_SimTk_;
  const edm::EDGetTokenT<edm::SimVertexContainer> tok_SimVtx_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloEB_, tok_caloEE_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloHH_;

  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;

  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB_, actHE_;

  const HcalDDDRecConstants* hcons_;

  static const int depthMax_ = 7;
  const int idMuon_ = 13;
  TTree* tree_;
  unsigned int runNumber_, eventNumber_, lumiNumber_, bxNumber_;
  double ptGlob_, etaGlob_, phiGlob_, pMuon_;
  double ecal3x3Energy_, hcal1x1Energy_;
  unsigned int ecalDetId_, hcalDetId_, hcalHot_;
  double matchedId_;
  double hcalDepthEnergy_[depthMax_];
  double hcalDepthActiveLength_[depthMax_];
  double hcalDepthEnergyHot_[depthMax_];
  double hcalDepthActiveLengthHot_[depthMax_];
  double hcalActiveLength_, hcalActiveLengthHot_;
};

HcalHBHEMuonSimAnalyzer::HcalHBHEMuonSimAnalyzer(const edm::ParameterSet& iConfig)
    : g4Label_(iConfig.getParameter<std::string>("ModuleLabel")),
      ebLabel_(iConfig.getParameter<std::string>("EBCollection")),
      eeLabel_(iConfig.getParameter<std::string>("EECollection")),
      hcLabel_(iConfig.getParameter<std::string>("HCCollection")),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      maxDepth_(iConfig.getUntrackedParameter<int>("MaxDepth", 7)),
      etaMax_(iConfig.getUntrackedParameter<double>("EtaMax", 3.0)),
      tMinE_(iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.)),
      tMaxE_(iConfig.getUntrackedParameter<double>("TimeMaxCutECAL", 500.)),
      tMinH_(iConfig.getUntrackedParameter<double>("TimeMinCutHCAL", -500.)),
      tMaxH_(iConfig.getUntrackedParameter<double>("TimeMaxCutHCAL", 500.)),
      tok_SimTk_(consumes<edm::SimTrackContainer>(edm::InputTag(g4Label_))),
      tok_SimVtx_(consumes<edm::SimVertexContainer>(edm::InputTag(g4Label_))),
      tok_caloEB_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, ebLabel_))),
      tok_caloEE_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, eeLabel_))),
      tok_caloHH_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hcLabel_))),
      tok_ddrec_(esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      tok_caloTopology_(esConsumes<CaloTopology, CaloTopologyRecord>()),
      tok_topo_(esConsumes<HcalTopology, HcalRecNumberingRecord>()),
      tok_magField_(esConsumes<MagneticField, IdealMagneticFieldRecord>()) {
  //now do what ever initialization is needed
  usesResource(TFileService::kSharedResource);

  edm::LogVerbatim("HBHEMuon") << "Labels: " << g4Label_ << ":" << ebLabel_ << ":" << eeLabel_ << ":" << hcLabel_
                               << "\nVerbosity " << verbosity_ << " MaxDepth " << maxDepth_ << " Maximum Eta "
                               << etaMax_ << " tMin|tMax " << tMinE_ << ":" << tMaxE_ << ":" << tMinH_ << ":" << tMaxH_;
}

void HcalHBHEMuonSimAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  clearVectors();
  bool debug(false);
#ifdef EDM_ML_DEBUG
  debug = ((verbosity_ / 10) > 0);
#endif

  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_ = iEvent.id().luminosityBlock();
  bxNumber_ = iEvent.bunchCrossing();

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk = iEvent.getHandle(tok_SimTk_);
  edm::Handle<edm::SimVertexContainer> SimVtx = iEvent.getHandle(tok_SimVtx_);

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb = iEvent.getHandle(tok_caloEB_);
  edm::Handle<edm::PCaloHitContainer> pcaloee = iEvent.getHandle(tok_caloEE_);
  edm::Handle<edm::PCaloHitContainer> pcalohh = iEvent.getHandle(tok_caloHH_);
  std::vector<PCaloHit> calohh;
  bool testN(false);
  for (unsigned int k = 1; k < pcalohh->size(); ++k) {
    // if it is a standard DetId bits 28..31 will carry the det #
    // for HCAL det # is 4 and if there is at least one hit in the collection
    // have det # which is not 4 this collection is created using TestNumbering
    int det = ((((*pcalohh)[k].id()) >> 28) & 0xF);
    if (det != 4) {
      testN = true;
      break;
    }
  }
  if (testN) {
    for (auto hit : (*(pcalohh.product()))) {
      DetId newid = HcalHitRelabeller::relabel(hit.id(), hcons_);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HBHEMuon") << "Old ID " << std::hex << hit.id() << std::dec << " New " << HcalDetId(newid);
#endif
      hit.setID(newid.rawId());
      calohh.push_back(hit);
    }
  } else {
    calohh.insert(calohh.end(), pcalohh->begin(), pcalohh->end());
  }

  // get handles to calogeometry and calotopology
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const MagneticField* bField = &iSetup.getData(tok_magField_);
  const CaloTopology* caloTopology = &iSetup.getData(tok_caloTopology_);
  const HcalTopology* theHBHETopology = &iSetup.getData(tok_topo_);

  std::vector<double> ptGlob, etaGlob, phiGlob, pMuon;
  std::vector<double> ecal3x3Energy, hcal1x1Energy;
  std::vector<unsigned int> ecalDetId, hcalDetId, hcalHot;
  std::vector<double> matchedId;
  std::vector<double> hcalDepthEnergy[depthMax_];
  std::vector<double> hcalDepthActiveLength[depthMax_];
  std::vector<double> hcalDepthEnergyHot[depthMax_];
  std::vector<double> hcalDepthActiveLengthHot[depthMax_];
  std::vector<double> hcalActiveLength, hcalActiveLengthHot;

  // Loop over all SimTracks
  for (const auto& simTrkItr : (*(SimTk.product()))) {
    if ((std::abs(simTrkItr.type()) == idMuon_) && (simTrkItr.vertIndex() == 0) &&
        (std::abs(simTrkItr.momentum().eta()) < etaMax_)) {
      unsigned int thisTrk = simTrkItr.trackId();
      spr::propagatedTrackDirection trkD = spr::propagateCALO(thisTrk, SimTk, SimVtx, geo, bField, debug);

      double eEcal(0), eHcal(0), activeLengthHotTot(0);
      double eHcalDepth[depthMax_], eHcalDepthHot[depthMax_];
      double activeL[depthMax_], activeHotL[depthMax_];
      unsigned int isHot(0);
      bool tmpmatch(false);
      for (int i = 0; i < depthMax_; ++i)
        eHcalDepth[i] = eHcalDepthHot[i] = activeL[i] = activeHotL[i] = -10000;

#ifdef EDM_ML_DEBUG
      if ((verbosity_ % 10) > 0)
        edm::LogVerbatim("HBHEMuon") << "Track Type " << simTrkItr.type() << " Vertex " << simTrkItr.vertIndex()
                                     << " Charge " << simTrkItr.charge() << " Momentum " << simTrkItr.momentum().P()
                                     << ":" << simTrkItr.momentum().eta() << ":" << simTrkItr.momentum().phi()
                                     << " ECAL|HCAL " << trkD.okECAL << ":" << trkD.okHCAL << " Point "
                                     << trkD.pointECAL << ":" << trkD.pointHCAL << " Direction "
                                     << trkD.directionECAL.eta() << ":" << trkD.directionECAL.phi() << " | "
                                     << trkD.directionHCAL.eta() << ":" << trkD.directionHCAL.phi();
#endif
      bool propageback(false);
      spr::propagatedTrackDirection trkD_back = spr::propagateHCALBack(thisTrk, SimTk, SimVtx, geo, bField, debug);
      HcalDetId closestCell_back;
      if (trkD_back.okHCAL) {
        closestCell_back = (HcalDetId)(trkD_back.detIdHCAL);
        propageback = true;
      }
      if (trkD.okHCAL) {
        // Muon properties
        spr::trackAtOrigin tkvx = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
        ptGlob.push_back(tkvx.momentum.perp());
        etaGlob.push_back(tkvx.momentum.eta());
        phiGlob.push_back(tkvx.momentum.phi());
        pMuon.push_back(tkvx.momentum.mag());
#ifdef EDM_ML_DEBUG
        if ((verbosity_ % 10) > 0)
          edm::LogVerbatim("HBHEMuon") << "Track at vertex " << tkvx.ok << " position " << tkvx.position << " Momentum "
                                       << tkvx.momentum.mag() << ":" << tkvx.momentum.eta() << ":"
                                       << tkvx.momentum.phi() << " Charge " << tkvx.charge;
#endif
        // Energy in ECAL
        DetId isoCell;
        if (trkD.okECAL) {
          isoCell = trkD.detIdECAL;
          eEcal = spr::eECALmatrix(
              isoCell, pcaloeb, pcaloee, geo, caloTopology, 1, 1, -100.0, -100.0, tMinE_, tMaxE_, debug);
        }

        // Energy in  Hcal
        const DetId closestCell(trkD.detIdHCAL);
        if ((propageback) && (HcalDetId(closestCell).ieta() == HcalDetId(closestCell_back).ieta()) &&
            (HcalDetId(closestCell).iphi() == HcalDetId(closestCell_back).iphi()))
          tmpmatch = true;

        eHcal = spr::eHCALmatrix(
            theHBHETopology, closestCell, calohh, 0, 0, false, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_, debug);
#ifdef EDM_ML_DEBUG
        if ((verbosity_ % 10) > 0)
          edm::LogVerbatim("HBHEMuon") << "eEcal " << trkD.okECAL << ":" << eEcal << " eHcal " << eHcal;
#endif
        HcalSubdetector subdet = HcalDetId(closestCell).subdet();
        int ieta = HcalDetId(closestCell).ieta();
        int iphi = HcalDetId(closestCell).iphi();
        int zside = HcalDetId(closestCell).zside();
        bool hbhe = (std::abs(ieta) == 16);
        int depthHE = hcons_->getMinDepth(1, 16, iphi, zside);
        std::vector<std::pair<double, int> > ehdepth;
        spr::energyHCALCell((HcalDetId)closestCell,
                            calohh,
                            ehdepth,
                            maxDepth_,
                            -100.0,
                            -100.0,
                            -100.0,
                            -100.0,
                            -500.0,
                            500.0,
                            depthHE,
                            debug);
        for (unsigned int i = 0; i < ehdepth.size(); ++i) {
          eHcalDepth[ehdepth[i].second - 1] = ehdepth[i].first;
          HcalSubdetector subdet0 = (hbhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
          HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
          double actL = activeLength(DetId(hcid0));
          activeL[ehdepth[i].second - 1] = actL;
#ifdef EDM_ML_DEBUG
          if ((verbosity_ % 10) > 0)
            edm::LogVerbatim("HBHEMuon") << hcid0 << " E " << ehdepth[i].first << " L " << actL;
#endif
        }

        HcalDetId hotCell;
        double h3x3 = spr::eHCALmatrix(geo, theHBHETopology, closestCell, calohh, 1, 1, hotCell, debug);
        isHot = matchId(closestCell, hotCell);
        if ((verbosity_ % 10) > 0)
          edm::LogVerbatim("HBHEMuon") << "hcal 3X3  < " << h3x3 << "> ClosestCell <" << (HcalDetId)(closestCell)
                                       << "> hotCell id < " << hotCell << "> isHot" << isHot;
        if (hotCell != HcalDetId()) {
          subdet = HcalDetId(hotCell).subdet();
          ieta = HcalDetId(hotCell).ieta();
          iphi = HcalDetId(hotCell).iphi();
          zside = HcalDetId(hotCell).zside();
          hbhe = (std::abs(ieta) == 16);
          depthHE = hcons_->getMinDepth(1, 16, iphi, zside);
          std::vector<std::pair<double, int> > ehdepth;
          spr::energyHCALCell(
              hotCell, calohh, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_, depthHE, debug);
          for (unsigned int i = 0; i < ehdepth.size(); ++i) {
            eHcalDepthHot[ehdepth[i].second - 1] = ehdepth[i].first;
            HcalSubdetector subdet0 = (hbhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
            HcalDetId hcid0(subdet0, ieta, iphi, ehdepth[i].second);
            double actL = activeLength(DetId(hcid0));
            activeHotL[ehdepth[i].second - 1] = actL;
            activeLengthHotTot += actL;
#ifdef EDM_ML_DEBUG
            if ((verbosity_ % 10) > 0)
              edm::LogVerbatim("HBHEMuon") << hcid0 << " E " << ehdepth[i].first << " L " << actL;
#endif
          }
        }
#ifdef EDM_ML_DEBUG
        if ((verbosity_ % 10) > 0) {
          for (int k = 0; k < depthMax_; ++k)
            edm::LogVerbatim("HBHEMuon") << "Depth " << k << " E " << eHcalDepth[k] << ":" << eHcalDepthHot[k];
        }
#endif
        matchedId.push_back(tmpmatch);
        ecal3x3Energy.push_back(eEcal);
        ecalDetId.push_back(isoCell.rawId());
        hcal1x1Energy.push_back(eHcal);
        hcalDetId.push_back(closestCell.rawId());
        for (int k = 0; k < depthMax_; ++k) {
          hcalDepthEnergy[k].push_back(eHcalDepth[k]);
          hcalDepthActiveLength[k].push_back(activeL[k]);
          hcalDepthEnergyHot[k].push_back(eHcalDepthHot[k]);
          hcalDepthActiveLengthHot[k].push_back(activeHotL[k]);
        }
        hcalHot.push_back(isHot);
        hcalActiveLengthHot.push_back(activeLengthHotTot);
      }
    }
  }
  for (unsigned int k = 0; k < hcalHot.size(); ++k) {
    ptGlob_ = ptGlob[k];
    etaGlob_ = etaGlob[k];
    phiGlob_ = phiGlob[k];
    pMuon_ = pMuon[k];
    ecal3x3Energy_ = ecal3x3Energy[k];
    hcal1x1Energy_ = hcal1x1Energy[k];
    ecalDetId_ = ecalDetId[k];
    hcalDetId_ = hcalDetId[k];
    hcalHot_ = hcalHot[k];
    matchedId_ = matchedId[k];
    for (int i = 0; i < depthMax_; ++i) {
      hcalDepthEnergy_[i] = hcalDepthEnergy[k][i];
      hcalDepthActiveLength_[i] = hcalDepthActiveLength[k][i];
      hcalDepthEnergyHot_[i] = hcalDepthEnergyHot[k][i];
      hcalDepthActiveLengthHot_[i] = hcalDepthActiveLengthHot[k][i];
    }
    hcalActiveLength_ = hcalActiveLength[k];
    hcalActiveLengthHot_ = hcalActiveLengthHot[k];
    tree_->Fill();
  }
}

void HcalHBHEMuonSimAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("TREE", "TREE");
  tree_->Branch("Run_No", &runNumber_);
  tree_->Branch("Event_No", &eventNumber_);
  tree_->Branch("LumiNumber", &lumiNumber_);
  tree_->Branch("BXNumber", &bxNumber_);
  tree_->Branch("pt_of_muon", &ptGlob_);
  tree_->Branch("eta_of_muon", &etaGlob_);
  tree_->Branch("phi_of_muon", &phiGlob_);
  tree_->Branch("p_of_muon", &pMuon_);
  tree_->Branch("matchedId", &matchedId_);

  tree_->Branch("ecal_3x3", &ecal3x3Energy_);
  tree_->Branch("ecal_detID", &ecalDetId_);
  tree_->Branch("hcal_1x1", &hcal1x1Energy_);
  tree_->Branch("hcal_detID", &hcalDetId_);
  tree_->Branch("hcal_cellHot", &hcalHot_);
  tree_->Branch("activeLength", &hcalActiveLength_);
  tree_->Branch("activeLengthHot", &hcalActiveLengthHot_);
  char name[100];
  for (int k = 0; k < maxDepth_; ++k) {
    sprintf(name, "hcal_edepth%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergy_[k]);
    sprintf(name, "hcal_activeL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLength_[k]);
    sprintf(name, "hcal_edepthHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyHot_[k]);
    sprintf(name, "hcal_activeHotL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLength_[k]);
  }
}

void HcalHBHEMuonSimAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  hcons_ = &iSetup.getData(tok_ddrec_);
  actHB_.clear();
  actHE_.clear();
  actHB_ = hcons_->getThickActive(0);
  actHE_ = hcons_->getThickActive(1);
}

void HcalHBHEMuonSimAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::string>("EBCollection", "EcalHitsEB");
  desc.add<std::string>("EECollection", "EcalHitsEE");
  desc.add<std::string>("HCCollection", "HcalHits");
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<int>("MaxDepth", 7);
  desc.addUntracked<double>("EtaMax", 3.0);
  desc.addUntracked<double>("TimeMinCutECAL", -500.);
  desc.addUntracked<double>("TimeMaxCutECAL", 500.);
  desc.addUntracked<double>("TimeMinCutHCAL", -500.);
  desc.addUntracked<double>("TimeMaxCutHCAL", 500.);
  descriptions.add("hcalHBHEMuonSim", desc);
}

void HcalHBHEMuonSimAnalyzer::clearVectors() {
  ///clearing vectots
  runNumber_ = -99999;
  eventNumber_ = -99999;
  lumiNumber_ = -99999;
  bxNumber_ = -99999;
}

unsigned int HcalHBHEMuonSimAnalyzer::matchId(const HcalDetId& id1, const HcalDetId& id2) {
  HcalDetId kd1(id1.subdet(), id1.ieta(), id1.iphi(), 1);
  HcalDetId kd2(id2.subdet(), id2.ieta(), id2.iphi(), 1);
  unsigned int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalHBHEMuonSimAnalyzer::activeLength(const DetId& id_) {
  HcalDetId id(id_);
  int ieta = id.ietaAbs();
  int depth = id.depth();
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    for (unsigned int i = 0; i < actHB_.size(); ++i) {
      if (ieta == actHB_[i].ieta && depth == actHB_[i].depth) {
        lx = actHB_[i].thick;
        break;
      }
    }
  } else {
    for (unsigned int i = 0; i < actHE_.size(); ++i) {
      if (ieta == actHE_[i].ieta && depth == actHE_[i].depth) {
        lx = actHE_[i].thick;
        break;
      }
    }
  }
  return lx;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHBHEMuonSimAnalyzer);
