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
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//#define EDM_ML_DEBUG

class HcalHBHEMuonSimAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HcalHBHEMuonSimAnalyzer(const edm::ParameterSet&);
  ~HcalHBHEMuonSimAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
   
private:
  virtual void beginJob() override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  void         clearVectors();
  unsigned int matchId(const HcalDetId&, const HcalDetId&);
  double       activeLength(const DetId&);

  std::string                                   g4Label_, ebLabel_, eeLabel_;
  std::string                                   hcLabel_;
  int                                           verbosity_, maxDepth_;
  double                                        etaMax_;
  std::vector<HcalDDDRecConstants::HcalActiveLength> actHB_, actHE_;
  const int                                     MaxDepth=7;
  const int                                     idMuon_=13;
  double                                        tMinE_, tMaxE_, tMinH_, tMaxH_;
  edm::Service<TFileService>                    fs_;
  edm::EDGetTokenT<edm::SimTrackContainer>      tok_SimTk_;
  edm::EDGetTokenT<edm::SimVertexContainer>     tok_SimVtx_;
  edm::EDGetTokenT<edm::PCaloHitContainer>      tok_caloEB_, tok_caloEE_;
  edm::EDGetTokenT<edm::PCaloHitContainer>      tok_caloHH_;

  TTree                    *tree_;
  unsigned int              runNumber_, eventNumber_, lumiNumber_, bxNumber_;
  std::vector<double>       ptGlob_, etaGlob_, phiGlob_, pMuon_;
  std::vector<double>       ecal3x3Energy_, hcal1x1Energy_;
  std::vector<unsigned int> ecalDetId_,     hcalDetId_,  hcalHot_;
  std::vector<double>       hcalDepth1Energy_, hcalDepth1ActiveLength_;
  std::vector<double>       hcalDepth2Energy_, hcalDepth2ActiveLength_;
  std::vector<double>       hcalDepth3Energy_, hcalDepth3ActiveLength_;
  std::vector<double>       hcalDepth4Energy_, hcalDepth4ActiveLength_;
  std::vector<double>       hcalDepth5Energy_, hcalDepth5ActiveLength_;
  std::vector<double>       hcalDepth6Energy_, hcalDepth6ActiveLength_;
  std::vector<double>       hcalDepth7Energy_, hcalDepth7ActiveLength_;
  std::vector<double>       hcalDepth1EnergyHot_, hcalDepth1ActiveLengthHot_;
  std::vector<double>       hcalDepth2EnergyHot_, hcalDepth2ActiveLengthHot_;
  std::vector<double>       hcalDepth3EnergyHot_, hcalDepth3ActiveLengthHot_;
  std::vector<double>       hcalDepth4EnergyHot_, hcalDepth4ActiveLengthHot_;
  std::vector<double>       hcalDepth5EnergyHot_, hcalDepth5ActiveLengthHot_;
  std::vector<double>       hcalDepth6EnergyHot_, hcalDepth6ActiveLengthHot_;
  std::vector<double>       hcalDepth7EnergyHot_, hcalDepth7ActiveLengthHot_;
  std::vector<double>       hcalActiveLength_,    hcalActiveLengthHot_;
};

HcalHBHEMuonSimAnalyzer::HcalHBHEMuonSimAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource(TFileService::kSharedResource);

  //now do what ever initialization is needed
  g4Label_   = iConfig.getParameter<std::string>("ModuleLabel");
  ebLabel_   = iConfig.getParameter<std::string>("EBCollection");
  eeLabel_   = iConfig.getParameter<std::string>("EECollection");
  hcLabel_   = iConfig.getParameter<std::string>("HCCollection");
  verbosity_ = iConfig.getUntrackedParameter<int>("Verbosity",0);
  maxDepth_  = iConfig.getUntrackedParameter<int>("MaxDepth",4);
  etaMax_    = iConfig.getUntrackedParameter<double>("EtaMax", 3.0);
  tMinE_     = iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.);
  tMaxE_     = iConfig.getUntrackedParameter<double>("TimeMaxCutECAL",  500.);
  tMinH_     = iConfig.getUntrackedParameter<double>("TimeMinCutHCAL", -500.);
  tMaxH_     = iConfig.getUntrackedParameter<double>("TimeMaxCutHCAL",  500.);

  tok_SimTk_  = consumes<edm::SimTrackContainer>(edm::InputTag(g4Label_));
  tok_SimVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag(g4Label_));
  tok_caloEB_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_,ebLabel_));
  tok_caloEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_,eeLabel_));
  tok_caloHH_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_,hcLabel_));
  if      (maxDepth_ > MaxDepth) maxDepth_ = MaxDepth;
  else if (maxDepth_ < 1)        maxDepth_ = 4;

  std::cout << "Labels: " << g4Label_ << ":" << ebLabel_ << ":" << eeLabel_
	    << ":" << hcLabel_ << "\nVerbosity " << verbosity_ << " MaxDepth "
	    << maxDepth_ << " Maximum Eta " << etaMax_ << " tMin|tMax "
	    << tMinE_ << ":" << tMaxE_ << ":" << tMinH_ << ":" << tMaxH_
	    << std::endl;
}

HcalHBHEMuonSimAnalyzer::~HcalHBHEMuonSimAnalyzer() { }

void HcalHBHEMuonSimAnalyzer::analyze(const edm::Event& iEvent,
				      const edm::EventSetup& iSetup) {
  
  clearVectors();
  bool debug(false);
#ifdef EDM_ML_DEBUG
  debug = ((verbosity_/10)>0);
#endif
  // depthHE is the first depth index for HE for |ieta| = 16
  // It used to be 3 for all runs preceding 2017 and 4 beyond that
  int depthHE = (maxDepth_ <= 6) ? 3 : 4;

  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  const HcalDDDRecConstants* hcons = &(*pHRNDC);

  runNumber_   = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_  = iEvent.id().luminosityBlock();
  bxNumber_    = iEvent.bunchCrossing();

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByToken(tok_SimTk_,SimTk);
  edm::SimTrackContainer::const_iterator simTrkItr;
  edm::Handle<edm::SimVertexContainer> SimVtx;
  iEvent.getByToken(tok_SimVtx_,SimVtx);

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  iEvent.getByToken(tok_caloEB_, pcaloeb);
  edm::Handle<edm::PCaloHitContainer> pcaloee;
  iEvent.getByToken(tok_caloEE_, pcaloee);
  edm::Handle<edm::PCaloHitContainer> pcalohh;
  iEvent.getByToken(tok_caloHH_, pcalohh);
  std::vector<PCaloHit> calohh;
  bool testN(false);
  for (unsigned int k=1; k<pcalohh->size(); ++k) {
    // if it is a standard DetId bits 28..31 will carry the det #
    // for HCAL det # is 4 and if there is at least one hit in the collection
    // have det # which is not 4 this collection is created using TestNumbering
    int det = ((((*pcalohh)[k].id())>>28)&0xF);
    if (det != 4) {testN = true; break;}
  }
  if (testN) {
    for (edm::PCaloHitContainer::const_iterator itr=pcalohh->begin(); itr != pcalohh->end(); ++itr) {
      PCaloHit hit(*itr);
      DetId newid = HcalHitRelabeller::relabel(hit.id(),hcons);
      std::cout << "Old ID " << std::hex << hit.id() << std::dec << " New " << HcalDetId(newid) << std::endl;
      hit.setID(newid.rawId());
      calohh.push_back(hit);
    }
  } else {
    calohh.insert(calohh.end(),pcalohh->begin(),pcalohh->end());
  }

  // get handles to calogeometry and calotopology
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  
  edm::ESHandle<MagneticField> bFieldH;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField* bField = bFieldH.product();

  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  const CaloTopology *caloTopology = theCaloTopology.product();
  
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* theHBHETopology = htopo.product();

  // Loop over all SimTracks
  for (edm::SimTrackContainer::const_iterator simTrkItr=SimTk->begin();
       simTrkItr!= SimTk->end(); simTrkItr++) {
    if ((std::abs(simTrkItr->type()) == idMuon_) && (simTrkItr->vertIndex() == 0) &&
	(std::abs(simTrkItr->momentum().eta()) < etaMax_)) {
      unsigned int thisTrk = simTrkItr->trackId();
      spr::propagatedTrackDirection trkD = spr::propagateCALO(thisTrk, SimTk, SimVtx, geo, bField, debug);

      double eEcal(0), eHcal(0), activeLengthTot(0), activeLengthHotTot(0);
      double eHcalDepth[MaxDepth], eHcalDepthHot[MaxDepth];
      double activeL[MaxDepth], activeHotL[MaxDepth];
      unsigned int isHot(0);
      for (int i=0; i<MaxDepth; ++i) 
	eHcalDepth[i] = eHcalDepthHot[i] = activeL[i] = activeHotL[i] = -10000;

#ifdef EDM_ML_DEBUG
      if ((verbosity_%10) > 0)
	std::cout << "Track Type " << simTrkItr->type() << " Vertex "
		  << simTrkItr->vertIndex() << " Charge " << simTrkItr->charge()
		  << " Momentum " << simTrkItr->momentum().P() << ":"
		  << simTrkItr->momentum().eta() << ":"
		  << simTrkItr->momentum().phi() << " ECAL|HCAL " << trkD.okECAL
		  << ":" << trkD.okHCAL << " Point " << trkD.pointECAL << ":"
		  << trkD.pointHCAL << " Direction " << trkD.directionECAL.eta()
		  << ":" << trkD.directionECAL.phi() << " | "
		  << trkD.directionHCAL.eta() << ":" << trkD.directionHCAL.phi()
		  << std::endl;
#endif

      if (trkD.okHCAL) {
	// Muon properties
	spr::trackAtOrigin tkvx = spr::simTrackAtOrigin(thisTrk, SimTk, SimVtx, debug);
	ptGlob_.push_back(tkvx.momentum.perp());
	etaGlob_.push_back(tkvx.momentum.eta());
	phiGlob_.push_back(tkvx.momentum.phi());
	pMuon_.push_back(tkvx.momentum.mag());
#ifdef EDM_ML_DEBUG
	if ((verbosity_%10) > 0)
	  std::cout << "Track at vertex " << tkvx.ok << " position " 
		    << tkvx.position << " Momentum " << tkvx.momentum.mag()
		    << ":" << tkvx.momentum.eta() << ":"
		    << tkvx.momentum.phi() << " Charge " << tkvx.charge
		    << std::endl;
#endif

	// Energy in ECAL
	DetId isoCell;
	if (trkD.okECAL) {
	  isoCell = trkD.detIdECAL;
	  eEcal = spr::eECALmatrix(isoCell, pcaloeb, pcaloee, geo, caloTopology, 1, 1, -100.0, -100.0, tMinE_, tMaxE_, debug);
	}

	// Energy in  Hcal
	const DetId closestCell(trkD.detIdHCAL);
	eHcal = spr::eHCALmatrix(theHBHETopology, closestCell, calohh,0,0, false, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_, debug);
#ifdef EDM_ML_DEBUG
	if ((verbosity_%10) > 0)
	  std::cout << "eEcal " << trkD.okECAL << ":" << eEcal << " eHcal " 
		    << eHcal << std::endl;
#endif

	HcalSubdetector subdet = HcalDetId(closestCell).subdet();
	int             ieta   = HcalDetId(closestCell).ieta();
	int             iphi   = HcalDetId(closestCell).iphi();
	bool            hbhe   = (std::abs(ieta) == 16);
	std::vector<std::pair<double,int> > ehdepth;
	spr::energyHCALCell((HcalDetId)closestCell, calohh, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, -500.0, 500.0, debug);
	for (unsigned int i=0; i<ehdepth.size(); ++i) {
	  eHcalDepth[ehdepth[i].second-1] = ehdepth[i].first;
	  HcalSubdetector subdet0 = (hbhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	  HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	  double actL = activeLength(DetId(hcid0));
	  activeL[ehdepth[i].second-1] = actL;
	  activeLengthTot += actL;
#ifdef EDM_ML_DEBUG
	  if ((verbosity_%10) > 0)
	    std::cout << hcid0 << " E " << ehdepth[i].first << " L " << actL 
		      << std::endl;
#endif
	}

	HcalDetId hotCell;
#ifdef EDM_ML_DEBUG
	double h3x3 = 
#endif
	  spr::eHCALmatrix(geo,theHBHETopology, closestCell, calohh, 1,1, hotCell, debug);
	isHot = matchId(closestCell,hotCell);
#ifdef EDM_ML_DEBUG
	if ((verbosity_%10) > 0)
	  std::cout << "hcal 3X3  < " << h3x3 << ">" << " ClosestCell <" 
		    << (HcalDetId)(closestCell) << "> hotCell id < " << hotCell
		    << "> isHot" << isHot << std::endl;
#endif

	if (hotCell != HcalDetId()) {
	  subdet = HcalDetId(hotCell).subdet();
	  ieta   = HcalDetId(hotCell).ieta();
	  iphi   = HcalDetId(hotCell).iphi();
	  hbhe   = (std::abs(ieta) == 16);
	  std::vector<std::pair<double,int> > ehdepth;
	  spr::energyHCALCell(hotCell, calohh, ehdepth, maxDepth_, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_, debug);
	  for (unsigned int i=0; i<ehdepth.size(); ++i) {
	    eHcalDepthHot[ehdepth[i].second-1] = ehdepth[i].first;
	    HcalSubdetector subdet0 = (hbhe) ? ((ehdepth[i].second >= depthHE) ? HcalEndcap : HcalBarrel) : subdet;
	    HcalDetId hcid0(subdet0,ieta,iphi,ehdepth[i].second);
	    double actL = activeLength(DetId(hcid0));
	    activeHotL[ehdepth[i].second-1] = actL;
	    activeLengthHotTot += actL;
#ifdef EDM_ML_DEBUG
	      if ((verbosity_%10) > 0)
		std::cout << hcid0 << " E " << ehdepth[i].first << " L " 
			  << actL << std::endl;
#endif
	  }
	}
#ifdef EDM_ML_DEBUG
	if ((verbosity_%10) > 0) {
	  for (int k=0; k<MaxDepth; ++k)
	    std::cout << "Depth " << k << " E " << eHcalDepth[k] << ":" 
		      << eHcalDepthHot[k] << std::endl;
	}
#endif      
	ecal3x3Energy_.push_back(eEcal);
	ecalDetId_.push_back(isoCell.rawId());
	hcal1x1Energy_.push_back(eHcal);
	hcalDetId_.push_back(closestCell.rawId());
	hcalDepth1Energy_.push_back(eHcalDepth[0]);
	hcalDepth2Energy_.push_back(eHcalDepth[1]);
	hcalDepth3Energy_.push_back(eHcalDepth[2]);
	hcalDepth4Energy_.push_back(eHcalDepth[3]);
	hcalDepth5Energy_.push_back(eHcalDepth[4]);
	hcalDepth6Energy_.push_back(eHcalDepth[5]);
	hcalDepth7Energy_.push_back(eHcalDepth[6]);
	hcalDepth1ActiveLength_.push_back(activeL[0]);
	hcalDepth2ActiveLength_.push_back(activeL[1]);
	hcalDepth3ActiveLength_.push_back(activeL[2]);
	hcalDepth4ActiveLength_.push_back(activeL[3]);
	hcalDepth5ActiveLength_.push_back(activeL[4]);
	hcalDepth6ActiveLength_.push_back(activeL[5]);
	hcalDepth7ActiveLength_.push_back(activeL[6]);
	hcalActiveLength_.push_back(activeLengthTot);
	hcalHot_.push_back(isHot);
	hcalDepth1EnergyHot_.push_back(eHcalDepthHot[0]);
	hcalDepth2EnergyHot_.push_back(eHcalDepthHot[1]);
	hcalDepth3EnergyHot_.push_back(eHcalDepthHot[2]);
	hcalDepth4EnergyHot_.push_back(eHcalDepthHot[3]);
	hcalDepth5EnergyHot_.push_back(eHcalDepthHot[4]);
	hcalDepth6EnergyHot_.push_back(eHcalDepthHot[5]);
	hcalDepth7EnergyHot_.push_back(eHcalDepthHot[6]);
	hcalDepth1ActiveLengthHot_.push_back(activeHotL[0]);
	hcalDepth2ActiveLengthHot_.push_back(activeHotL[1]);
	hcalDepth3ActiveLengthHot_.push_back(activeHotL[2]);
	hcalDepth4ActiveLengthHot_.push_back(activeHotL[3]);
	hcalDepth5ActiveLengthHot_.push_back(activeHotL[4]);
	hcalDepth6ActiveLengthHot_.push_back(activeHotL[5]);
	hcalDepth7ActiveLengthHot_.push_back(activeHotL[6]);
	hcalActiveLengthHot_.push_back(activeLengthHotTot);
      }
    }
  }
  if (hcalHot_.size() > 0) tree_->Fill();
}

void HcalHBHEMuonSimAnalyzer::beginJob() {

  tree_ = fs_->make<TTree>("TREE", "TREE");
  tree_->Branch("Run_No",           &runNumber_);
  tree_->Branch("Event_No",         &eventNumber_);
  tree_->Branch("LumiNumber",       &lumiNumber_);
  tree_->Branch("BXNumber",         &bxNumber_);
  tree_->Branch("pt_of_muon",       &ptGlob_);
  tree_->Branch("eta_of_muon",      &etaGlob_);
  tree_->Branch("phi_of_muon",      &phiGlob_);
  tree_->Branch("p_of_muon",        &pMuon_);
  
  tree_->Branch("ecal_3x3",         &ecal3x3Energy_);
  tree_->Branch("ecal_detID",       &ecalDetId_);
  tree_->Branch("hcal_1x1",         &hcal1x1Energy_);
  tree_->Branch("hcal_detID",       &hcalDetId_);
  tree_->Branch("hcal_cellHot",     &hcalHot_);
  tree_->Branch("activeLength",     &hcalActiveLength_);
  tree_->Branch("hcal_edepth1",     &hcalDepth1Energy_);
  tree_->Branch("hcal_edepth2",     &hcalDepth2Energy_);
  tree_->Branch("hcal_edepth3",     &hcalDepth3Energy_);
  tree_->Branch("hcal_edepth4",     &hcalDepth4Energy_);
  tree_->Branch("hcal_activeL1",    &hcalDepth1ActiveLength_);
  tree_->Branch("hcal_activeL2",    &hcalDepth2ActiveLength_);
  tree_->Branch("hcal_activeL3",    &hcalDepth3ActiveLength_);
  tree_->Branch("hcal_activeL4",    &hcalDepth4ActiveLength_);
  tree_->Branch("activeLengthHot",  &hcalActiveLengthHot_);
  tree_->Branch("hcal_edepthHot1",  &hcalDepth1EnergyHot_);
  tree_->Branch("hcal_edepthHot2",  &hcalDepth2EnergyHot_);
  tree_->Branch("hcal_edepthHot3",  &hcalDepth3EnergyHot_);
  tree_->Branch("hcal_edepthHot4",  &hcalDepth4EnergyHot_);
  tree_->Branch("hcal_activeHotL1", &hcalDepth1ActiveLength_);
  tree_->Branch("hcal_activeHotL2", &hcalDepth2ActiveLength_);
  tree_->Branch("hcal_activeHotL3", &hcalDepth3ActiveLength_);
  tree_->Branch("hcal_activeHotL4", &hcalDepth4ActiveLength_);
  if (maxDepth_ > 4) {
    tree_->Branch("hcal_edepth5",     &hcalDepth5Energy_);
    tree_->Branch("hcal_activeL5",    &hcalDepth5ActiveLength_);
    tree_->Branch("hcal_edepthHot5",  &hcalDepth5EnergyHot_);
    tree_->Branch("hcal_activeHotL5", &hcalDepth5ActiveLength_);
    if (maxDepth_ > 5) {
      tree_->Branch("hcal_edepth6",     &hcalDepth6Energy_);
      tree_->Branch("hcal_activeL6",    &hcalDepth6ActiveLength_);
      tree_->Branch("hcal_edepthHot6",  &hcalDepth6EnergyHot_);
      tree_->Branch("hcal_activeHotL6", &hcalDepth6ActiveLength_);
      if (maxDepth_ > 6) {
	tree_->Branch("hcal_edepth7",     &hcalDepth7Energy_);
	tree_->Branch("hcal_activeL7",    &hcalDepth7ActiveLength_);
	tree_->Branch("hcal_edepthHot7",  &hcalDepth7EnergyHot_);
	tree_->Branch("hcal_activeHotL7", &hcalDepth7ActiveLength_);
      }
    }
  }

}

void HcalHBHEMuonSimAnalyzer::beginRun(edm::Run const& iRun, 
				       edm::EventSetup const& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);
  const HcalDDDRecConstants & hdc = (*pHRNDC);
  actHB_.clear();
  actHE_.clear();
  actHB_ = hdc.getThickActive(0);
  actHE_ = hdc.getThickActive(1);
}


void HcalHBHEMuonSimAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel","g4SimHits");
  desc.add<std::string>("EBCollection","EcalHitsEB");
  desc.add<std::string>("EECollection","EcalHitsEE");
  desc.add<std::string>("HCCollection","HcalHits");
  desc.addUntracked<int>("Verbosity",0);
  desc.addUntracked<int>("MaxDepth",4);
  desc.addUntracked<double>("EtaMax",3.0);
  desc.addUntracked<double>("TimeMinCutECAL",-500.0);
  desc.addUntracked<double>("TimeMaxCutECAL",500.0);
  desc.addUntracked<double>("TimeMinCutHCAL",-500.0);
  desc.addUntracked<double>("TimeMaxCutHCAL",500.0);
  descriptions.add("hcalHBHEMuonSim",desc);
}

void HcalHBHEMuonSimAnalyzer::clearVectors() {

  ///clearing vectots
  runNumber_   = -99999;
  eventNumber_ = -99999;
  lumiNumber_  = -99999;
  bxNumber_    = -99999;

  ptGlob_.clear();
  etaGlob_.clear(); 
  phiGlob_.clear(); 
  pMuon_.clear();

  ecal3x3Energy_.clear();
  ecalDetId_.clear();
  hcal1x1Energy_.clear();
  hcalDetId_.clear();
  hcalHot_.clear();
  hcalActiveLength_.clear();
  hcalDepth1Energy_.clear();
  hcalDepth2Energy_.clear();
  hcalDepth3Energy_.clear();
  hcalDepth4Energy_.clear();
  hcalDepth5Energy_.clear();
  hcalDepth6Energy_.clear();
  hcalDepth7Energy_.clear();
  hcalDepth1ActiveLength_.clear();
  hcalDepth2ActiveLength_.clear();
  hcalDepth3ActiveLength_.clear();
  hcalDepth4ActiveLength_.clear();
  hcalDepth5ActiveLength_.clear();
  hcalDepth6ActiveLength_.clear();
  hcalDepth7ActiveLength_.clear();
  hcalActiveLengthHot_.clear();
  hcalDepth1EnergyHot_.clear();
  hcalDepth2EnergyHot_.clear();
  hcalDepth3EnergyHot_.clear();
  hcalDepth4EnergyHot_.clear();
  hcalDepth5EnergyHot_.clear();
  hcalDepth6EnergyHot_.clear();
  hcalDepth7EnergyHot_.clear();
  hcalDepth1ActiveLengthHot_.clear();
  hcalDepth2ActiveLengthHot_.clear();
  hcalDepth3ActiveLengthHot_.clear();
  hcalDepth4ActiveLengthHot_.clear();
  hcalDepth5ActiveLengthHot_.clear();
  hcalDepth6ActiveLengthHot_.clear();
  hcalDepth7ActiveLengthHot_.clear();
}

unsigned int HcalHBHEMuonSimAnalyzer::matchId(const HcalDetId& id1, 
					      const HcalDetId& id2) {

  HcalDetId kd1(id1.subdet(),id1.ieta(),id1.iphi(),1);
  HcalDetId kd2(id2.subdet(),id2.ieta(),id2.iphi(),1);
  unsigned int match = ((kd1 == kd2) ? 1 : 0);
  return match;
}

double HcalHBHEMuonSimAnalyzer::activeLength(const DetId& id_) {
  HcalDetId id(id_);
  int ieta = id.ietaAbs();
  int depth= id.depth();
  double lx(0);
  if (id.subdet() == HcalBarrel) {
    for (unsigned int i=0; i<actHB_.size(); ++i) {
      if (ieta == actHB_[i].ieta && depth == actHB_[i].depth) {
	lx = actHB_[i].thick;
	break;
      }
    }
  } else {
    for (unsigned int i=0; i<actHE_.size(); ++i) {
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
