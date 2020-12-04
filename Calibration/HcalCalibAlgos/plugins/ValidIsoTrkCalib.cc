//
// Package:    ValidIsoTrkCalib
// Class:      ValidIsoTrkCalib
//

/*
 Description: Validation for Isolated tracks Calibration
 
 Implementation:
See the twiki page for details:
https://twiki.cern.ch/twiki/bin/view/CMS/ValidIsoTrkCalib

*/

//
// Original Author:  Andrey Pozdnyakov
//         Created:  Tue Nov  4 01:16:05 CET 2008
//

// system include files

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "Calibration/HcalCalibAlgos/interface/CommonUsefulStuff.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"

#include <fstream>
#include <map>

class ValidIsoTrkCalib : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ValidIsoTrkCalib(const edm::ParameterSet&);

  //  double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  //Variables from HcalIsotrackAnalyzer

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;
  double taECALCone_;
  double taHCALCone_;

  const CaloGeometry* geo;
  // nothing is done with these tags, so I leave it - cowden
  edm::InputTag genhbheLabel_;
  edm::InputTag genhoLabel_;
  std::vector<edm::InputTag> genecalLabel_;

  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> tok_track_;
  edm::EDGetTokenT<reco::TrackCollection> tok_track1_;

  edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_recalibCorrs_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;

  //std::string m_inputTrackLabel;
  //std::string m_hcalLabel;

  double associationConeSize_;
  string AxB_;
  double calibrationConeSize_;

  bool allowMissingInputs_;
  string outputFileName_;
  //string calibFactorsFileName_;
  //  string corrfile;

  bool takeGenTracks_;
  //bool takeAllRecHits_, takeGenTracks_;
  int gen, iso, pix;
  float genPt[500], genPhi[500], genEta[500];
  float isoPt[500], isoPhi[500], isoEta[500];
  float pixPt[500], pixPhi[500], pixEta[500];
  //int  hbheiEta[5000],hbheiPhi[5000],hbheDepth[5000];
  //float hbheEnergy[5000];

  int NisoTrk;
  float trackPt, trackE, trackEta, trackPhi;
  float ptNear;
  float ptrack, rvert;
  //float eecal, ehcal;

  int MinNTrackHitsBarrel, MinNTECHitsEndcap;
  double energyECALmip, maxPNear;
  double energyMinIso, energyMaxIso;

  Float_t emEnergy;
  Float_t targetE;

  //TFile* rootFile;
  //  TTree* tree;
  TTree *tTree, *fTree;

  Float_t xTrkEcal;
  Float_t yTrkEcal;
  Float_t zTrkEcal;

  Float_t xTrkHcal;
  Float_t yTrkHcal;
  Float_t zTrkHcal;

  int Nhits;
  float eClustBefore;  //Calo energy before calibration
  float eClustAfter;   //After calibration
  float eTrack;        //Track energy
  float etaTrack;
  float phiTrack;
  float eECAL;  // Energy deposited in ECAL
  int numHits;  //number of rechits
  //float e3x3;

  float eBeforeDepth1;
  float eAfterDepth1;
  float eBeforeDepth2;
  float eAfterDepth2;
  float eCentHitBefore;
  float eCentHitAfter;
  float CentHitFactor;
  int iEta;
  int iPhi;
  int iEtaTr;
  int iPhiTr;
  float iDr, delR;
  int dietatr;
  int diphitr;

  float iTime;
  float HTime[100];
  float e3x3Before;
  float e3x3After;
  float e5x5Before;
  float e5x5After;
  int eventNumber;
  int runNumber;
  float PtNearBy;
  float numVH, numVS, numValidTrkHits, numValidTrkStrips;

  const HcalRespCorrs* respRecalib;
  //  map<UInt_t, Float_t> CalibFactors;
  //  Bool_t ReadCalibFactors(string);

  TH1F* nTracks;

  edm::Service<TFileService> fs;
  // int Lumi_n;
};

ValidIsoTrkCalib::ValidIsoTrkCalib(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);

  //takeAllRecHits_=iConfig.getUntrackedParameter<bool>("takeAllRecHits");
  takeGenTracks_ = iConfig.getUntrackedParameter<bool>("takeGenTracks");

  tok_genTrack_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("genTracksLabel"));
  genhbheLabel_ = iConfig.getParameter<edm::InputTag>("genHBHE");
  //genhoLabel_=iConfig.getParameter<edm::InputTag>("genHO");
  //genecalLabel_=iConfig.getParameter<std::vector<edm::InputTag> >("genECAL");

  // m_hcalLabel = iConfig.getUntrackedParameter<std::string> ("hcalRecHitsLabel","hbhereco");

  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  //eLabel_=iConfig.getParameter<edm::InputTag>("eInput");
  tok_track_ =
      consumes<reco::IsolatedPixelTrackCandidateCollection>(iConfig.getParameter<edm::InputTag>("HcalIsolTrackInput"));
  tok_track1_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackInput"));

  tok_recalibCorrs_ = esConsumes(edm::ESInputTag("", "recalibrate"));
  tok_geom_ = esConsumes();

  associationConeSize_ = iConfig.getParameter<double>("associationConeSize");
  allowMissingInputs_ = iConfig.getUntrackedParameter<bool>("allowMissingInputs", true);
  //  outputFileName_=iConfig.getParameter<std::string>("outputFileName");
  //  calibFactorsFileName_=iConfig.getParameter<std::string>("calibFactorsFileName");

  AxB_ = iConfig.getParameter<std::string>("AxB");
  calibrationConeSize_ = iConfig.getParameter<double>("calibrationConeSize");

  MinNTrackHitsBarrel = iConfig.getParameter<int>("MinNTrackHitsBarrel");
  MinNTECHitsEndcap = iConfig.getParameter<int>("MinNTECHitsEndcap");
  energyECALmip = iConfig.getParameter<double>("energyECALmip");
  energyMinIso = iConfig.getParameter<double>("energyMinIso");
  energyMaxIso = iConfig.getParameter<double>("energyMaxIso");
  maxPNear = iConfig.getParameter<double>("maxPNear");

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);
  trackAssociator_.useDefaultPropagator();

  // taECALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  //taHCALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorHCALCone",0.6);
}

// ------------ method called to for each event  ------------
void ValidIsoTrkCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  try {
    respRecalib = &iSetup.getData(tok_recalibCorrs_);

    edm::LogInfo("CalibConstants") << "  Loaded:  OK ";

  } catch (const cms::Exception& e) {
    edm::LogWarning("CalibConstants") << "   Not Found!! ";
  }

  edm::Handle<reco::TrackCollection> generalTracks;
  iEvent.getByToken(tok_genTrack_, generalTracks);

  edm::Handle<reco::TrackCollection> isoProdTracks;
  iEvent.getByToken(tok_track1_, isoProdTracks);

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isoPixelTracks;
  //edm::Handle<reco::TrackCollection> isoPixelTracks;
  iEvent.getByToken(tok_track_, isoPixelTracks);

  /*
  edm::Handle<EcalRecHitCollection> ecal;
  iEvent.getByLabel(eLabel_,ecal);
  const EcalRecHitCollection Hitecal = *(ecal.product());
  */

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  geo = &iSetup.getData(tok_geom_);

  const HcalGeometry* gHcal = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  //Note: even though it says HcalBarrel, we actually get the whole Hcal detector geometry!

  // Lumi_n=iEvent.luminosityBlock();
  parameters_.useEcal = true;
  parameters_.useHcal = true;
  parameters_.useCalo = false;
  parameters_.useMuon = false;
  //parameters_.dREcal = taECALCone_;
  //parameters_.dRHcal = taHCALCone_;

  //cout<<"Hello World. TrackCollectionSize: "<< isoPixelTracks->size()<<endl;

  if (isoPixelTracks->empty())
    return;

  for (reco::TrackCollection::const_iterator trit = isoProdTracks->begin(); trit != isoProdTracks->end(); trit++) {
    reco::IsolatedPixelTrackCandidateCollection::const_iterator isoMatched = isoPixelTracks->begin();
    //reco::TrackCollection::const_iterator isoMatched=isoPixelTracks->begin();
    bool matched = false;

    //for (reco::IsolatedPixelTrackCandidateCollection::const_iterator trit = isoPixelTracks->begin(); trit!=isoPixelTracks->end(); trit++)
    for (reco::IsolatedPixelTrackCandidateCollection::const_iterator it = isoPixelTracks->begin();
         it != isoPixelTracks->end();
         it++)
    //for (reco::TrackCollection::const_iterator it = isoPixelTracks->begin(); it!=isoPixelTracks->end(); it++)
    {
      if (abs((trit->pt() - it->pt()) / it->pt()) < 0.005 && abs(trit->eta() - it->eta()) < 0.01) {
        isoMatched = it;
        matched = true;
        break;
      }
    }
    // CUT

    if (!matched)
      continue;
    if (isoMatched->maxPtPxl() > maxPNear)
      continue;

    ptNear = isoMatched->maxPtPxl();
    //cout<<"Point 0.1  isoMatch. ptnear: "<<ptNear<<endl;

    // CUT
    if (trit->hitPattern().numberOfValidHits() < MinNTrackHitsBarrel)
      continue;
    if (fabs(trit->eta()) > 1.47 && trit->hitPattern().numberOfValidStripTECHits() < MinNTECHitsEndcap)
      continue;

    //cout<<"Point 0.2.1 after numofvalidhits HB: "<<trit->hitPattern().numberOfValidHits()<<endl;
    //cout<<"Point 0.2.2 after numofvalidstrips HE: "<<trit->hitPattern().numberOfValidStripTECHits()<<endl;

    numVH = trit->hitPattern().numberOfValidHits();
    numVS = trit->hitPattern().numberOfValidStripTECHits();

    trackE = sqrt(trit->px() * trit->px() + trit->py() * trit->py() + trit->pz() * trit->pz() + 0.14 * 0.14);
    trackPt = trit->pt();
    trackEta = trit->eta();
    trackPhi = trit->phi();

    emEnergy = isoMatched->energyIn();

    //cout<<"Point 0.3.  Matched :: pt: "<<trit->pt()<<" wholeEnergy: "<<trackE<<"  emEnergy: "<<emEnergy<<"  eta: "<<etahcal<<" phi: "<<phihcal<<endl;
    //cout<<"Point 0.4.  EM energy in cone: "<<emEnergy<<"  EtaHcal: "<<etahcal<<"  PhiHcal: "<<phihcal<<endl;

    TrackDetMatchInfo info = trackAssociator_.associate(
        iEvent,
        iSetup,
        trackAssociator_.getFreeTrajectoryState(&iSetup.getData(parameters_.bFieldToken), *trit),
        parameters_);

    //float etaecal=info.trkGlobPosAtEcal.eta();
    //float phiecal=info.trkGlobPosAtEcal.phi();
    //  float etahcal=info.trkGlobPosAtHcal.eta();
    // float phihcal=info.trkGlobPosAtHcal.phi();

    xTrkEcal = info.trkGlobPosAtEcal.x();
    yTrkEcal = info.trkGlobPosAtEcal.y();
    zTrkEcal = info.trkGlobPosAtEcal.z();

    xTrkHcal = info.trkGlobPosAtHcal.x();
    yTrkHcal = info.trkGlobPosAtHcal.y();
    zTrkHcal = info.trkGlobPosAtHcal.z();

    if (xTrkEcal == 0 && yTrkEcal == 0 && zTrkEcal == 0) {
      cout << "zero point at Ecal" << endl;
      continue;
    }
    if (xTrkHcal == 0 && yTrkHcal == 0 && zTrkHcal == 0) {
      cout << "zero point at Hcal" << endl;
      continue;
    }

    /*GlobalVector trackMomAtEcal = info.trkMomAtEcal;
      GlobalVector trackMomAtHcal = info.trkMomAtHcal;
	
      PxTrkHcal = trackMomAtHcal.x();
      PyTrkHcal = trackMomAtHcal.y();
      PzTrkHcal = trackMomAtHcal.z();
      */

    GlobalPoint gPointEcal(xTrkEcal, yTrkEcal, zTrkEcal);
    GlobalPoint gPointHcal(xTrkHcal, yTrkHcal, zTrkHcal);

    int iphitrue = -10;
    int ietatrue = 100;
    const HcalDetId tempId = gHcal->getClosestCell(gPointHcal);
    ietatrue = tempId.ieta();
    iphitrue = tempId.iphi();

    MaxHit_struct MaxHit;

    MaxHit.hitenergy = -100.;

    //container for used recHits
    std::vector<DetId> usedHits;
    //
    usedHits.clear();
    //cout <<"Point 1. Entrance to HBHECollection"<<endl;

    //float dddeta = 1000.;
    //float dddphi = 1000.;
    //int iphitrue = 1234;
    //int ietatrue = 1234;

    GlobalPoint gPhot;

    for (HBHERecHitCollection::const_iterator hhit = Hithbhe.begin(); hhit != Hithbhe.end(); hhit++) {
      //check that this hit was not considered before and push it into usedHits
      bool hitIsUsed = false;
      for (uint32_t i = 0; i < usedHits.size(); i++) {
        if (usedHits[i] == hhit->id())
          hitIsUsed = true;
      }
      if (hitIsUsed)
        continue;
      usedHits.push_back(hhit->id());
      //

      // rof 16.05.2008 start: include the possibility for recalibration
      float recal = 1;
      // rof end

      GlobalPoint pos = geo->getPosition(hhit->detid());
      //float phihit = pos.phi();
      //float etahit = pos.eta();

      int iphihitm = (hhit->id()).iphi();
      int ietahitm = (hhit->id()).ieta();
      int depthhit = (hhit->id()).depth();
      float enehit = hhit->energy() * recal;

      if (depthhit != 1)
        continue;

      /*
	    float dphi = fabs(phihcal - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    float deta = fabs(etahcal - etahit); 
	    float dr = sqrt(dphi*dphi + deta*deta);
	  */

      //double distAtHcal =  getDistInPlaneTrackDir(gPointHcal, trackMomAtHcal, pos);
      double distAtHcal = getDistInPlaneSimple(gPointHcal, pos);

      if (distAtHcal < associationConeSize_) {
        for (HBHERecHitCollection::const_iterator hhit2 = Hithbhe.begin(); hhit2 != Hithbhe.end(); hhit2++) {
          int iphihitm2 = (hhit2->id()).iphi();
          int ietahitm2 = (hhit2->id()).ieta();
          int depthhit2 = (hhit2->id()).depth();
          float enehit2 = hhit2->energy() * recal;

          if (iphihitm == iphihitm2 && ietahitm == ietahitm2 && depthhit != depthhit2)
            enehit = enehit + enehit2;
        }

        //cout<<"IN CONE ieta: "<<ietahitm<<"  iphi: "<<iphihitm<<" depthhit: "<<depthhit<<"  dr: "<<dr<<" energy: "<<enehit<<endl;

        //Find a Hit with Maximum Energy

        if (enehit > MaxHit.hitenergy) {
          MaxHit.hitenergy = enehit;
          MaxHit.ietahitm = (hhit->id()).ieta();
          MaxHit.iphihitm = (hhit->id()).iphi();
          MaxHit.dr = distAtHcal;
          //MaxHit.depthhit  = (hhit->id()).depth();
          MaxHit.depthhit = 1;

          //gPhot = geo->getPosition(hhit->detid());
        }
      }
    }  //end of all HBHE hits cycle

    usedHits.clear();

    //cout<<"Hottest ieta: "<<MaxHit.ietahitm<<"  iphi: "<<MaxHit.iphihitm<<"  dr: "<<MaxHit.dr<<endl;
    //cout<<"Track   ieta: "<<ietatrue<<"  iphi: "<<iphitrue<<endl;

    //cout<<"Point 3.  MaxHit :::En "<<MaxHit.hitenergy<<"  ieta: "<<MaxHit.ietahitm<<"  iphi: "<<MaxHit.iphihitm<<endl;

    Bool_t passCuts = kFALSE;
    if (trackE > energyMinIso && trackE < energyMaxIso && emEnergy < energyECALmip && MaxHit.hitenergy > 0. &&
        abs(MaxHit.ietahitm) < 29)
      passCuts = kTRUE;

    //cout<<"Pont 0.1.1.  trackE:"<<trackE<<"  emEn: "<<emEnergy<<endl;

    numHits = 0;

    eClustBefore = 0.0;
    eClustAfter = 0.0;
    eBeforeDepth1 = 0.0;
    eAfterDepth1 = 0.0;
    eBeforeDepth2 = 0.0;
    eAfterDepth2 = 0.0;
    CentHitFactor = 0.0;
    e3x3After = 0.0;
    e3x3Before = 0.0;
    e5x5After = 0.0;
    e5x5Before = 0.0;

    for (HBHERecHitCollection::const_iterator hhit = Hithbhe.begin(); hhit != Hithbhe.end(); hhit++) {
      //check that this hit was not considered before and push it into usedHits
      bool hitIsUsed = false;
      for (uint32_t i = 0; i < usedHits.size(); i++) {
        if (usedHits[i] == hhit->id())
          hitIsUsed = true;
      }
      if (hitIsUsed)
        continue;
      usedHits.push_back(hhit->id());

      int DIETA = 100;
      if (MaxHit.ietahitm * (hhit->id()).ieta() > 0) {
        DIETA = MaxHit.ietahitm - (hhit->id()).ieta();
      }
      if (MaxHit.ietahitm * (hhit->id()).ieta() < 0) {
        DIETA = MaxHit.ietahitm - (hhit->id()).ieta();
        DIETA = DIETA > 0 ? DIETA - 1 : DIETA + 1;
      }

      int DIPHI = abs(MaxHit.iphihitm - (hhit->id()).iphi());
      DIPHI = DIPHI > 36 ? 72 - DIPHI : DIPHI;

      int numbercell = 100;  //always collect Wide clastor!

      //if(AxB_=="5x5" || AxB_=="3x3" || AxB_=="7x7"|| AxB_=="Cone")

      //if(AxB_=="3x3") numbercell = 1;
      //if(AxB_=="5x5") numbercell = 2;
      //if(AxB_=="Cone") numbercell = 1000;

      if (abs(DIETA) <= numbercell &&
          (abs(DIPHI) <= numbercell || (abs(MaxHit.ietahitm) >= 20 && abs(DIPHI) <= numbercell + 1))) {
        const GlobalPoint pos2 = geo->getPosition(hhit->detid());

        if (passCuts && hhit->energy() > 0) {
          float factor = 0.0;
          // factor = CalibFactors[hhit->id()];
          factor = respRecalib->getValues(hhit->id())->getValue();

          //if(i<5){cout<<" calib factors: "<<factor<<"  ij "<<100*i+j<<endl;}

          if (hhit->id().ieta() == MaxHit.ietahitm && hhit->id().iphi() == MaxHit.iphihitm)
            CentHitFactor = factor;

          if (hhit->id().ieta() == MaxHit.ietahitm && hhit->id().iphi() == MaxHit.iphihitm)
            iTime = hhit->time();

          if (AxB_ != "3x3" && AxB_ != "5x5" && AxB_ != "Cone")
            edm::LogWarning(" AxB ") << "   Not supported: " << AxB_;

          if (abs(DIETA) <= 2 && (abs(DIPHI) <= 2 || ((abs(MaxHit.ietahitm) > 20 && abs(DIPHI) <= 4) &&
                                                      !((abs(MaxHit.ietahitm) == 21 || abs(MaxHit.ietahitm) == 22) &&
                                                        abs((hhit->id()).ieta()) <= 20 && abs(DIPHI) > 2)))) {
            e5x5Before += hhit->energy();
            e5x5After += hhit->energy() * factor;
          }

          if (abs(DIETA) <= 1 && (abs(DIPHI) <= 1 || ((abs(MaxHit.ietahitm) > 20 && abs(DIPHI) <= 2) &&
                                                      !(abs(MaxHit.ietahitm) == 21 && abs((hhit->id()).ieta()) <= 20 &&
                                                        abs(DIPHI) > 1)))) {
            e3x3Before += hhit->energy();
            e3x3After += hhit->energy() * factor;
          }

          if (AxB_ == "5x5") {
            if (abs(DIETA) <= 2 && (abs(DIPHI) <= 2 || (abs(MaxHit.ietahitm) > 20 && abs(DIPHI) <= 4))) {
              if (abs(MaxHit.ietahitm) == 21 && abs((hhit->id()).ieta()) <= 20 && abs(DIPHI) > 3)
                continue;

              HTime[numHits] = hhit->time();
              numHits++;

              eClustBefore += hhit->energy();
              eClustAfter += hhit->energy() * factor;

              if ((hhit->id().depth() == 1) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
                eBeforeDepth1 += hhit->energy();
                eAfterDepth1 += hhit->energy() * factor;
              } else if ((hhit->id().depth() == 2) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
                eBeforeDepth2 += hhit->energy();
                eAfterDepth2 += hhit->energy() * factor;
              }
            }
          }  //end of 5x5

          if (AxB_ == "3x3") {
            if (abs(DIETA) <= 1 && (abs(DIPHI) <= 1 || (abs(MaxHit.ietahitm) > 20 && abs(DIPHI) <= 2))) {
              if (abs(MaxHit.ietahitm) == 21 && abs((hhit->id()).ieta()) <= 20 && abs(DIPHI) > 2)
                continue;

              HTime[numHits] = hhit->time();
              numHits++;

              eClustBefore += hhit->energy();
              eClustAfter += hhit->energy() * factor;

              if ((hhit->id().depth() == 1) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
                eBeforeDepth1 += hhit->energy();
                eAfterDepth1 += hhit->energy() * factor;
              } else if ((hhit->id().depth() == 2) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
                eBeforeDepth2 += hhit->energy();
                eAfterDepth2 += hhit->energy() * factor;
              }
            }
          }  //end of 3x3

          if (AxB_ == "Cone" && getDistInPlaneSimple(gPointHcal, pos2) < calibrationConeSize_) {
            HTime[numHits] = hhit->time();
            numHits++;

            eClustBefore += hhit->energy();
            eClustAfter += hhit->energy() * factor;

            if ((hhit->id().depth() == 1) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
              eBeforeDepth1 += hhit->energy();
              eAfterDepth1 += hhit->energy() * factor;
            } else if ((hhit->id().depth() == 2) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29)) {
              eBeforeDepth2 += hhit->energy();
              eAfterDepth2 += hhit->energy() * factor;
            }

          }  //end of Cone

        }  //end of passCuts

      }  //end of DIETA DIPHI

    }  //end of associatedcone HBHE hits cycle

    int dieta_M_P = 100;
    int diphi_M_P = 100;
    if (MaxHit.ietahitm * ietatrue > 0) {
      dieta_M_P = abs(MaxHit.ietahitm - ietatrue);
    }
    if (MaxHit.ietahitm * ietatrue < 0) {
      dieta_M_P = abs(MaxHit.ietahitm - ietatrue) - 1;
    }
    diphi_M_P = abs(MaxHit.iphihitm - iphitrue);
    diphi_M_P = diphi_M_P > 36 ? 72 - diphi_M_P : diphi_M_P;

    if (passCuts)

    {
      eventNumber = iEvent.id().event();
      runNumber = iEvent.id().run();

      eCentHitBefore = MaxHit.hitenergy;
      eCentHitAfter = MaxHit.hitenergy * CentHitFactor;
      eECAL = emEnergy;
      numValidTrkHits = numVH;
      numValidTrkStrips = numVS;
      PtNearBy = ptNear;

      eTrack = trackE;
      phiTrack = trackPhi;
      etaTrack = trackEta;

      iEta = MaxHit.ietahitm;
      iPhi = MaxHit.iphihitm;

      iEtaTr = ietatrue;
      iPhiTr = iphitrue;
      iDr = sqrt(diphi_M_P * diphi_M_P + dieta_M_P * dieta_M_P);
      delR = MaxHit.dr;
      dietatr = dieta_M_P;
      diphitr = diphi_M_P;

      fTree->Fill();
    }

  }  //end of isoProdTracks cycle

  /* ------------------   Some stuff for general tracks  ----------   ----*/
  //cout<<" generalTracks Size: "<< generalTracks->size()<<endl;
  int n = generalTracks->size();
  nTracks->Fill(n);

  if (takeGenTracks_ && iEvent.id().event() % 10 == 1) {
    gen = generalTracks->size();
    iso = isoProdTracks->size();
    pix = isoPixelTracks->size();

    genPt[0] = -33;
    genPhi[0] = -33;
    genEta[0] = -33;

    isoPt[0] = -33;
    isoPhi[0] = -33;
    isoEta[0] = -33;

    pixPt[0] = -33;
    pixPhi[0] = -33;
    pixEta[0] = -33;

    Int_t gencount = 0, isocount = 0, pixcount = 0;
    for (reco::TrackCollection::const_iterator gentr = generalTracks->begin(); gentr != generalTracks->end(); gentr++) {
      genPt[gencount] = gentr->pt();
      genPhi[gencount] = gentr->phi();
      genEta[gencount] = gentr->eta();
      gencount++;
    }

    for (reco::TrackCollection::const_iterator isotr = isoProdTracks->begin(); isotr != isoProdTracks->end(); isotr++) {
      isoPt[isocount] = isotr->pt();
      isoPhi[isocount] = isotr->phi();
      isoEta[isocount] = isotr->eta();
      isocount++;
    }

    for (reco::IsolatedPixelTrackCandidateCollection::const_iterator pixtr = isoPixelTracks->begin();
         pixtr != isoPixelTracks->end();
         pixtr++) {
      pixPt[pixcount] = pixtr->pt();
      pixPhi[pixcount] = pixtr->phi();
      pixEta[pixcount] = pixtr->eta();
      pixcount++;
    }
  }

  tTree->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void ValidIsoTrkCalib::beginJob() {
  // if(!ReadCalibFactors(calibFactorsFileName_.c_str() )) {cout<<"Cant read file with cailib coefficients!! ---"<<endl;}

  //  rootFile = new TFile(outputFileName_.c_str(),"RECREATE");

  //@@@@@@@@@@@@@
  //TFileDirectory ValDir = fs->mkdir("Validation");

  nTracks = fs->make<TH1F>("nTracks", "general;number of general tracks", 11, -0.5, 10.5);

  tTree = fs->make<TTree>("tTree", "Tree for gen info");

  fTree = fs->make<TTree>("fTree", "Tree for IsoTrack Calibration");

  fTree->Branch("eventNumber", &eventNumber, "eventNumber/I");
  fTree->Branch("runNumber", &runNumber, "runNumber/I");

  fTree->Branch("eClustBefore", &eClustBefore, "eClustBefore/F");
  fTree->Branch("eClustAfter", &eClustAfter, "eClustAfter/F");
  fTree->Branch("eTrack", &eTrack, "eTrack/F");
  fTree->Branch("etaTrack", &etaTrack, "etaTrack/F");
  fTree->Branch("phiTrack", &phiTrack, "phiTrack/F");

  fTree->Branch("numHits", &numHits, "numHits/I");
  fTree->Branch("eECAL", &eECAL, "eECAL/F");
  fTree->Branch("PtNearBy", &PtNearBy, "PtNearBy/F");
  fTree->Branch("numValidTrkHits", &numValidTrkHits, "numValidTrkHits/F");
  fTree->Branch("numValidTrkStrips", &numValidTrkStrips, "numValidTrkStrips/F");

  fTree->Branch("eBeforeDepth1", &eBeforeDepth1, "eBeforeDepth1/F");
  fTree->Branch("eBeforeDepth2", &eBeforeDepth2, "eBeforeDepth2/F");
  fTree->Branch("eAfterDepth1", &eAfterDepth1, "eAfterDepth1/F");
  fTree->Branch("eAfterDepth2", &eAfterDepth2, "eAfterDepth2/F");

  fTree->Branch("e3x3Before", &e3x3Before, "e3x3Before/F");
  fTree->Branch("e3x3After", &e3x3After, "e3x3After/F");
  fTree->Branch("e5x5Before", &e5x5Before, "e5x5Before/F");
  fTree->Branch("e5x5After", &e5x5After, "e5x5After/F");

  fTree->Branch("eCentHitAfter", &eCentHitAfter, "eCentHitAfter/F");
  fTree->Branch("eCentHitBefore", &eCentHitBefore, "eCentHitBefore/F");
  fTree->Branch("iEta", &iEta, "iEta/I");
  fTree->Branch("iPhi", &iPhi, "iPhi/I");

  fTree->Branch("iEtaTr", &iEtaTr, "iEtaTr/I");
  fTree->Branch("iPhiTr", &iPhiTr, "iPhiTr/I");
  fTree->Branch("dietatr", &dietatr, "dietatr/I");
  fTree->Branch("diphitr", &diphitr, "diphitr/I");
  fTree->Branch("iDr", &iDr, "iDr/F");
  fTree->Branch("delR", &delR, "delR/F");

  fTree->Branch("iTime", &iTime, "iTime/F");
  fTree->Branch("HTime", HTime, "HTime[numHits]/F");

  fTree->Branch("xTrkEcal", &xTrkEcal, "xTrkEcal/F");
  fTree->Branch("yTrkEcal", &yTrkEcal, "yTrkEcal/F");
  fTree->Branch("zTrkEcal", &zTrkEcal, "zTrkEcal/F");
  fTree->Branch("xTrkHcal", &xTrkHcal, "xTrkHcal/F");
  fTree->Branch("yTrkHcal", &yTrkHcal, "yTrkHcal/F");
  fTree->Branch("zTrkHcal", &zTrkHcal, "zTrkHcal/F");

  if (takeGenTracks_) {
    tTree->Branch("gen", &gen, "gen/I");
    tTree->Branch("iso", &iso, "iso/I");
    tTree->Branch("pix", &pix, "pix/I");
    tTree->Branch("genPt", genPt, "genPt[gen]/F");
    tTree->Branch("genPhi", genPhi, "genPhi[gen]/F");
    tTree->Branch("genEta", genEta, "genEta[gen]/F");

    tTree->Branch("isoPt", isoPt, "isoPt[iso]/F");
    tTree->Branch("isoPhi", isoPhi, "isoPhi[iso]/F");
    tTree->Branch("isoEta", isoEta, "isoEta[iso]/F");

    tTree->Branch("pixPt", pixPt, "pixPt[pix]/F");
    tTree->Branch("pixPhi", pixPhi, "pixPhi[pix]/F");
    tTree->Branch("pixEta", pixEta, "pixEta[pix]/F");
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ValidIsoTrkCalib);
