// -*- C++ -*-
//
// Package:    RecHitTreeProducer
// Class:      RecHitTreeProducer
//
/**\class RecHitTreeProducer RecHitTreeProducer.cc CmsHi/RecHitTreeProducer/src/RecHitTreeProducer.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
// Modified: Frank Ma, Yen-Jie Lee
//         Created:  Tue Sep  7 11:38:19 EDT 2010
// $Id: RecHitTreeProducer.cc,v 1.27 2013/01/22 16:36:27 yilmaz Exp $
//
//

#define versionTag "v1"
// system include files
#include <memory>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"


#include "TNtuple.h"

using namespace std;

#define MAXHITS 100000


struct MyRecHit{
  int depth[MAXHITS];
  int n;

  int ieta[MAXHITS];
  int iphi[MAXHITS];

  float e[MAXHITS];
  float eraw[MAXHITS];
  float et[MAXHITS];
  float eta[MAXHITS];
  float phi[MAXHITS];
  float perp[MAXHITS];
  float emEt[MAXHITS];
  float hadEt[MAXHITS];
  float chi2[MAXHITS];
  float eError[MAXHITS];

  uint32_t flags[MAXHITS];
  
  bool isjet[MAXHITS];
  float etVtx[MAXHITS];
  float etaVtx[MAXHITS];
  float phiVtx[MAXHITS];
  float perpVtx[MAXHITS];
  float emEtVtx[MAXHITS];
  float hadEtVtx[MAXHITS];

  int saturation[MAXHITS];

  float jtpt;
  float jteta;
  float jtphi;

  Float_t                vsPt[MAXHITS];
  Float_t                vsPtInitial[MAXHITS];
  Float_t                vsArea[MAXHITS];

  Float_t                 sumpt[20];
  Float_t                 vn[200];
  Float_t                 psin[200];
  Float_t                 ueraw[1200];

};


struct MyZDCRecHit{
  int n;
  float  e[18];
  int    zside[18];
  int    section [18];
  int    channel[18];
  int    saturation[18];
};

struct MyZDCDigi{
  int    n;
  float  chargefC[10][18];
  int    adc[10][18];
  int    zside[18];
  int    section [18];
  int    channel[18];
};

struct MyBkg{
  int n;
  float rho[50];
  float sigma[50];
};


//
// class declaration
//

class RecHitTreeProducer : public edm::EDAnalyzer {
public:
  explicit RecHitTreeProducer(const edm::ParameterSet&);
  ~RecHitTreeProducer();
  double       getEt(const DetId &id, double energy, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double       getEta(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double       getPhi(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double       getPerp(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));

  math::XYZPoint getPosition(const DetId &id, reco::Vertex::Point vtx = reco::Vertex::Point(0,0,0));
  double getEt(math::XYZPoint pos, double energy);

  reco::Vertex::Point getVtx(const edm::Event& ev);



private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::Handle<reco::Centrality> cent;
  edm::Handle<vector<double> > ktRhos;
  edm::Handle<vector<double> > akRhos;

  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > ebHits;
  edm::Handle<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > eeHits;

  edm::Handle<HFRecHitCollection> hfHits;
  edm::Handle<HBHERecHitCollection> hbheHits;

  edm::Handle<reco::BasicClusterCollection> bClusters;
  edm::Handle<CaloTowerCollection> towers;
  edm::Handle<reco::CandidateView> candidates_;

  edm::Handle<reco::VertexCollection> vtxs;

  typedef vector<EcalRecHit>::const_iterator EcalIterator;

  edm::Handle<reco::CaloJetCollection> jets;

  edm::Handle<std::vector<double> > rhos;
  edm::Handle<std::vector<double> > sigmas;

  edm::Handle<reco::VoronoiMap> backgrounds_;
  edm::Handle<std::vector<float> > vn_;

  MyRecHit hbheRecHit;
  MyRecHit hfRecHit;
  MyRecHit ebRecHit;
  MyRecHit eeRecHit;
  MyRecHit myBC;
  MyRecHit myTowers;
  MyRecHit castorRecHit;

  MyZDCRecHit zdcRecHit;
  MyZDCDigi zdcDigi;

  MyBkg bkg;

  TNtuple* nt;
  TTree* hbheTree;
  TTree* hfTree;
  TTree* ebTree;
  TTree* eeTree;
  TTree* bcTree;
  TTree* towerTree;
  TTree* bkgTree;
  TTree* castorTree;
  TTree* zdcRecHitTree;
  TTree* zdcDigiTree;

  double cone;
  double hfTowerThreshold_;
  double hfLongThreshold_;
  double hfShortThreshold_;
  double hbheThreshold_;
  double ebThreshold_;
  double eeThreshold_;

  double hbhePtMin_;
  double hfPtMin_;
  double ebPtMin_;
  double eePtMin_;
  double towerPtMin_;

  int nZdcTs_;
  bool calZDCDigi_;

  edm::Service<TFileService> fs;
  const CentralityBins * cbins_;
  const CaloGeometry *geo;

  edm::InputTag HcalRecHitHFSrc_;
  edm::InputTag HcalRecHitHBHESrc_;
  edm::InputTag zdcDigiSrc_;
  edm::InputTag zdcRecHitSrc_;

  edm::InputTag EBSrc_;
  edm::InputTag EESrc_;
  edm::InputTag BCSrc_;
  edm::InputTag TowerSrc_;
  edm::InputTag VtxSrc_;

  edm::InputTag JetSrc_;

  edm::InputTag FastJetTag_;

  edm::InputTag srcVor_;
  int           fourierOrder_;
  int           etaBins_;
  bool   doUEraw_;


  bool useJets_;
  bool doBasicClusters_;
  bool doTowers_;
  bool doEcal_;
  bool doHcal_;
  bool doHF_;
  bool doCastor_;
  bool doZDCRecHit_;
  bool doZDCDigi_;

  bool doVS_;

  bool hasVtx_;
  bool saveBothVtx_;

  bool doFastJet_;

  bool doEbyEonly_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RecHitTreeProducer::RecHitTreeProducer(const edm::ParameterSet& iConfig) :
  cone(0.5),
  cbins_(0),
  geo(0)
{
  //now do what ever initialization is needed
  EBSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("EBRecHitSrc",edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  EESrc_ = iConfig.getUntrackedParameter<edm::InputTag>("EERecHitSrc",edm::InputTag("ecalRecHit","EcalRecHitsEE"));
  HcalRecHitHFSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHFRecHitSrc",edm::InputTag("hfreco"));
  HcalRecHitHBHESrc_ = iConfig.getUntrackedParameter<edm::InputTag>("hcalHBHERecHitSrc",edm::InputTag("hbhereco"));
  zdcDigiSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("zdcDigiSrc",edm::InputTag("castorDigis"));
  zdcRecHitSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("zdcRecHitSrc",edm::InputTag("nothing"));

  BCSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("BasicClusterSrc1",edm::InputTag("ecalRecHit","EcalRecHitsEB","RECO"));
  TowerSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("towersSrc",edm::InputTag("towerMaker"));
  VtxSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("vtxSrc",edm::InputTag("hiSelectedVertex"));
  JetSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("JetSrc",edm::InputTag("iterativeConePu5CaloJets"));
  useJets_ = iConfig.getUntrackedParameter<bool>("useJets",true);
  doBasicClusters_ = iConfig.getUntrackedParameter<bool>("doBasicClusters",false);
  doTowers_ = iConfig.getUntrackedParameter<bool>("doTowers",true);
  doEcal_ = iConfig.getUntrackedParameter<bool>("doEcal",true);
  doHcal_ = iConfig.getUntrackedParameter<bool>("doHcal",true);
  doHF_ = iConfig.getUntrackedParameter<bool>("doHF",true);
  doCastor_ = iConfig.getUntrackedParameter<bool>("doCASTOR",true);
  doZDCRecHit_ = iConfig.getUntrackedParameter<bool>("doZDCRecHit",true);
  doZDCDigi_ = iConfig.getUntrackedParameter<bool>("doZDCDigi",true);
  doVS_ = iConfig.getUntrackedParameter<bool>("doVS",true);

  doUEraw_ = iConfig.getUntrackedParameter<bool>("doUEraw",0);

  etaBins_ = iConfig.getParameter<int>("etaBins");
  fourierOrder_ = iConfig.getParameter<int>("fourierOrder");

  srcVor_ = iConfig.getParameter<edm::InputTag>("bkg");

  hasVtx_ = iConfig.getUntrackedParameter<bool>("hasVtx",true);
  saveBothVtx_ = iConfig.getUntrackedParameter<bool>("saveBothVtx",false);

  doFastJet_ = iConfig.getUntrackedParameter<bool>("doFastJet",true);
  FastJetTag_ = iConfig.getUntrackedParameter<edm::InputTag>("FastJetTag",edm::InputTag("kt4CaloJets"));
  doEbyEonly_ = iConfig.getUntrackedParameter<bool>("doEbyEonly",false);
  hfTowerThreshold_ = iConfig.getUntrackedParameter<double>("HFtowerMin",3.);
  hfLongThreshold_ = iConfig.getUntrackedParameter<double>("HFlongMin",0.5);
  hfShortThreshold_ = iConfig.getUntrackedParameter<double>("HFshortMin",0.85);
  hbhePtMin_ = iConfig.getUntrackedParameter<double>("HBHETreePtMin",0);
  hfPtMin_ = iConfig.getUntrackedParameter<double>("HFTreePtMin",0);
  ebPtMin_ = iConfig.getUntrackedParameter<double>("EBTreePtMin",0);
  eePtMin_ = iConfig.getUntrackedParameter<double>("EETreePtMin",0.);
  towerPtMin_ = iConfig.getUntrackedParameter<double>("TowerTreePtMin",0.);
  nZdcTs_=iConfig.getUntrackedParameter<int>("nZdcTs",10);
  calZDCDigi_=iConfig.getUntrackedParameter<bool>("calZDCDigi",true);
}


RecHitTreeProducer::~RecHitTreeProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RecHitTreeProducer::analyze(const edm::Event& ev, const edm::EventSetup& iSetup)
{

  hfRecHit.n = 0;
  hbheRecHit.n = 0;
  ebRecHit.n = 0;
  eeRecHit.n = 0;
  myBC.n = 0;
  myTowers.n = 0;
  bkg.n = 0;

  // get vertex
  reco::Vertex::Point vtx(0,0,0);
  if (hasVtx_) vtx = getVtx(ev);

  if(doEcal_){
    ev.getByLabel(EBSrc_,ebHits);
    ev.getByLabel(EESrc_,eeHits);
  }

  if(doHcal_){
    ev.getByLabel(HcalRecHitHBHESrc_,hbheHits);
  }
  if(doHF_){
    ev.getByLabel(HcalRecHitHFSrc_,hfHits);
  }

  if(useJets_) {
    ev.getByLabel(JetSrc_,jets);
  }

  if(doBasicClusters_){
    ev.getByLabel(BCSrc_,bClusters);
  }

  if(doTowers_){
    ev.getByLabel(TowerSrc_,towers);

  }

  if (doTowers_ && doVS_) {
    ev.getByLabel(TowerSrc_,candidates_);
    ev.getByLabel(srcVor_,backgrounds_);
    ev.getByLabel(srcVor_,vn_);

    UEParameters vnUE(vn_.product(),fourierOrder_,etaBins_);
    const std::vector<float>& vue = vnUE.get_raw();
    for(int ieta = 0; ieta < etaBins_; ++ieta){
      myTowers.sumpt[ieta] = vnUE.get_sum_pt(ieta);
      for(int ifour = 0; ifour < fourierOrder_; ++ifour){
	myTowers.vn[ifour * etaBins_ + ieta] = vnUE.get_vn(ifour,ieta);
	myTowers.psin[ifour * etaBins_ + ieta] = vnUE.get_psin(ifour,ieta);
      }
    }


    for(int iue = 0; iue < etaBins_*fourierOrder_*2*3; ++iue){
      myTowers.ueraw[iue] = vue[iue];
    }

  }

  if(doFastJet_){
    ev.getByLabel(edm::InputTag(FastJetTag_.label(),"rhos",FastJetTag_.process()),rhos);
    ev.getByLabel(edm::InputTag(FastJetTag_.label(),"sigmas",FastJetTag_.process()),sigmas);
    bkg.n = rhos->size();
    for(unsigned int i = 0; i < rhos->size(); ++i){
      bkg.rho[i] = (*rhos)[i];
      bkg.sigma[i] = (*sigmas)[i];
    }
  }

  //if(0 && !cbins_) cbins_ = getCentralityBinsFromDB(iSetup);

  if(!geo){
    edm::ESHandle<CaloGeometry> pGeo;
    iSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }

  int nHFlongPlus = 0;
  int nHFshortPlus = 0;
  int nHFtowerPlus = 0;
  int nHFlongMinus = 0;
  int nHFshortMinus = 0;
  int nHFtowerMinus = 0;

  if(doHF_){
    for(unsigned int i = 0; i < hfHits->size(); ++i){
      const HFRecHit & hit= (*hfHits)[i];
      hfRecHit.e[hfRecHit.n] = hit.energy();
      math::XYZPoint pos = getPosition(hit.id(),vtx);

      if(!saveBothVtx_){
	hfRecHit.et[hfRecHit.n] = getEt(pos,hit.energy());
	hfRecHit.eta[hfRecHit.n] = pos.eta();
	hfRecHit.phi[hfRecHit.n] = pos.phi();
	hfRecHit.perp[hfRecHit.n] = pos.rho();
      }else{
	hfRecHit.et[hfRecHit.n] = getEt(hit.id(),hit.energy());
	hfRecHit.eta[hfRecHit.n] = getEta(hit.id());
	hfRecHit.phi[hfRecHit.n] = getPhi(hit.id());
	hfRecHit.perp[hfRecHit.n] = getPerp(hit.id());

	hfRecHit.etVtx[hfRecHit.n] = getEt(pos,hit.energy());
	hfRecHit.etaVtx[hfRecHit.n] = pos.eta();
	hfRecHit.phiVtx[hfRecHit.n] = pos.phi();
	hfRecHit.perpVtx[hfRecHit.n] = pos.rho();

      }

      hfRecHit.isjet[hfRecHit.n] = false;
      hfRecHit.depth[hfRecHit.n] = hit.id().depth();

      if(hit.id().ieta() > 0){
	if(hit.energy() > hfShortThreshold_ && hit.id().depth() != 1) nHFshortPlus++;
	if(hit.energy() > hfLongThreshold_ && hit.id().depth() == 1) nHFlongPlus++;
      }else{
	if(hit.energy() > hfShortThreshold_ && hit.id().depth() != 1) nHFshortMinus++;
	if(hit.energy() > hfLongThreshold_ && hit.id().depth() == 1) nHFlongMinus++;
      }

      if(useJets_){
	for(unsigned int j = 0 ; j < jets->size(); ++j){
	  const reco::Jet& jet = (*jets)[j];
	  double dr = reco::deltaR(hfRecHit.eta[hfRecHit.n],hfRecHit.phi[hfRecHit.n],jet.eta(),jet.phi());
	  if(dr < cone){ hfRecHit.isjet[hfRecHit.n] = true; }
	}
      }
      if (hfRecHit.et[hfRecHit.n]>=hfPtMin_) hfRecHit.n++;
    }

    if(doHcal_ && !doEbyEonly_){
      for(unsigned int i = 0; i < hbheHits->size(); ++i){
	const HBHERecHit & hit= (*hbheHits)[i];
	if (getEt(hit.id(),hit.energy())<hbhePtMin_) continue;

	hbheRecHit.e[hbheRecHit.n] = hit.energy();
	hbheRecHit.eraw[hbheRecHit.n] = hit.eraw();
	math::XYZPoint pos = getPosition(hit.id(),vtx);

	if(!saveBothVtx_){
	  hbheRecHit.et[hbheRecHit.n] = getEt(pos,hit.energy());
	  hbheRecHit.eta[hbheRecHit.n] = pos.eta();
	  hbheRecHit.phi[hbheRecHit.n] = pos.phi();
	  hbheRecHit.perp[hbheRecHit.n] = pos.rho();
	}else{
	  hbheRecHit.et[hbheRecHit.n] = getEt(hit.id(),hit.energy());
	  hbheRecHit.eta[hbheRecHit.n] = getEta(hit.id());
	  hbheRecHit.phi[hbheRecHit.n] = getPhi(hit.id());
	  hbheRecHit.perp[hbheRecHit.n] = getPerp(hit.id());

	  hbheRecHit.etVtx[hbheRecHit.n] = getEt(pos,hit.energy());
	  hbheRecHit.etaVtx[hbheRecHit.n] = pos.eta();
	  hbheRecHit.phiVtx[hbheRecHit.n] = pos.phi();
	  hbheRecHit.perpVtx[hbheRecHit.n] = pos.rho();
	}
        
	hbheRecHit.isjet[hbheRecHit.n] = false;
	hbheRecHit.depth[hbheRecHit.n] = hit.id().depth();

	if(useJets_){
	  for(unsigned int j = 0 ; j < jets->size(); ++j){
	    const reco::Jet& jet = (*jets)[j];
	    double dr = reco::deltaR(hbheRecHit.eta[hbheRecHit.n],hbheRecHit.phi[hbheRecHit.n],jet.eta(),jet.phi());
	    if(dr < cone){ hbheRecHit.isjet[hbheRecHit.n] = true; }
	  }
	}
	hbheRecHit.n++;
      }
    }
  }
  if(doEcal_ && !doEbyEonly_){
    for(unsigned int i = 0; i < ebHits->size(); ++i){
      const EcalRecHit & hit= (*ebHits)[i];
      if (getEt(hit.id(),hit.energy())<ebPtMin_) continue;

      ebRecHit.e[ebRecHit.n] = hit.energy();
      math::XYZPoint pos = getPosition(hit.id(),vtx);

      if(!saveBothVtx_){
	ebRecHit.et[ebRecHit.n] = getEt(pos,hit.energy());
	ebRecHit.eta[ebRecHit.n] = pos.eta();
	ebRecHit.phi[ebRecHit.n] = pos.phi();
	ebRecHit.perp[ebRecHit.n] = pos.rho();
      }else{
	ebRecHit.et[ebRecHit.n] = getEt(hit.id(),hit.energy());
	ebRecHit.eta[ebRecHit.n] = getEta(hit.id());
	ebRecHit.phi[ebRecHit.n] = getPhi(hit.id());
	ebRecHit.perp[ebRecHit.n] = getPerp(hit.id());
	ebRecHit.etVtx[ebRecHit.n] = getEt(pos,hit.energy());
	ebRecHit.etaVtx[ebRecHit.n] = pos.eta();
	ebRecHit.phiVtx[ebRecHit.n] = pos.phi();
	ebRecHit.perpVtx[ebRecHit.n] = pos.rho();
      }
      ebRecHit.chi2[ebRecHit.n] = hit.chi2();
      ebRecHit.eError[ebRecHit.n] = hit.energyError();

      ebRecHit.flags[ebRecHit.n] = 0;
      for (uint32_t f=0; f<32; ++f) {
        if (hit.checkFlag(f))
          ebRecHit.flags[ebRecHit.n] |= 1 << f;
      }
      
      ebRecHit.isjet[ebRecHit.n] = false;
      if(useJets_){
	for(unsigned int j = 0 ; j < jets->size(); ++j){
	  const reco::Jet& jet = (*jets)[j];
	  double dr = reco::deltaR(ebRecHit.eta[ebRecHit.n],ebRecHit.phi[ebRecHit.n],jet.eta(),jet.phi());
	  if(dr < cone){ ebRecHit.isjet[ebRecHit.n] = true; }
	}
      }
      ebRecHit.n++;
    }

    for(unsigned int i = 0; i < eeHits->size(); ++i){
      const EcalRecHit & hit= (*eeHits)[i];
      if (getEt(hit.id(),hit.energy())<eePtMin_) continue;

      eeRecHit.e[eeRecHit.n] = hit.energy();
      math::XYZPoint pos = getPosition(hit.id(),vtx);

      if(!saveBothVtx_){
	eeRecHit.et[eeRecHit.n] = getEt(pos,hit.energy());
	eeRecHit.eta[eeRecHit.n] = pos.eta();
	eeRecHit.phi[eeRecHit.n] = pos.phi();
	eeRecHit.perp[eeRecHit.n] = pos.rho();
      }else{
	eeRecHit.et[eeRecHit.n] = getEt(hit.id(),hit.energy());
	eeRecHit.eta[eeRecHit.n] = getEta(hit.id());
	eeRecHit.phi[eeRecHit.n] = getPhi(hit.id());
	eeRecHit.perp[eeRecHit.n] = getPerp(hit.id());
	eeRecHit.etVtx[eeRecHit.n] = getEt(pos,hit.energy());
	eeRecHit.etaVtx[eeRecHit.n] = pos.eta();
	eeRecHit.phiVtx[eeRecHit.n] = pos.phi();
	eeRecHit.perpVtx[eeRecHit.n] = pos.rho();
      }
      eeRecHit.chi2[eeRecHit.n] = hit.chi2();
      eeRecHit.eError[eeRecHit.n] = hit.energyError();
      
      eeRecHit.flags[eeRecHit.n] = 0;
      for (unsigned int f=0; f<32; ++f) {
        if (hit.checkFlag(f))
          eeRecHit.flags[eeRecHit.n] |= 1 << f;
      }

      eeRecHit.isjet[eeRecHit.n] = false;

      if(useJets_){
	for(unsigned int j = 0 ; j < jets->size(); ++j){
	  const reco::Jet& jet = (*jets)[j];
	  double dr = reco::deltaR(eeRecHit.eta[eeRecHit.n],eeRecHit.phi[eeRecHit.n],jet.eta(),jet.phi());
	  if(dr < cone){ eeRecHit.isjet[eeRecHit.n] = true; }
	}
      }
      eeRecHit.n++;
    }
  }

  if(doTowers_){

    for(unsigned int i = 0; i < towers->size(); ++i){
      const CaloTower & hit= (*towers)[i];
      if (getEt(hit.id(),hit.energy())<towerPtMin_) continue;

      if (doVS_) {

	reco::CandidateViewRef ref(candidates_,i);
	double vsPtInitial=-999, vsPt=-999, vsArea = -999;

	const reco::VoronoiBackground& voronoi = (*backgrounds_)[ref];
	vsPt = voronoi.pt();
	vsPtInitial = voronoi.pt_subtracted();
	vsArea = voronoi.area();
	myTowers.vsPt[myTowers.n] = vsPt;
	myTowers.vsPtInitial[myTowers.n] = vsPtInitial;
	myTowers.vsArea[myTowers.n] = vsArea;

      }
      myTowers.e[myTowers.n] = hit.energy();
      myTowers.et[myTowers.n] = hit.p4(vtx).Et();
      myTowers.eta[myTowers.n] = hit.p4(vtx).Eta();
      myTowers.phi[myTowers.n] = hit.p4(vtx).Phi();
      myTowers.emEt[myTowers.n] = hit.emEt(vtx);
      myTowers.hadEt[myTowers.n] = hit.hadEt(vtx);

      if (saveBothVtx_) {
	myTowers.e[myTowers.n] = hit.energy();
	myTowers.et[myTowers.n] = getEt(hit.id(),hit.energy());
	myTowers.eta[myTowers.n] = getEta(hit.id());
	myTowers.phi[myTowers.n] = getPhi(hit.id());
	myTowers.isjet[myTowers.n] = false;
	myTowers.etVtx[myTowers.n] = hit.p4(vtx).Et();
	myTowers.etaVtx[myTowers.n] = hit.p4(vtx).Eta();
	myTowers.emEtVtx[myTowers.n] = hit.emEt(vtx);
	myTowers.hadEtVtx[myTowers.n] = hit.hadEt(vtx);
      }

      myTowers.isjet[myTowers.n] = false;

      if(hit.ieta() > 29 && hit.energy() > hfTowerThreshold_) nHFtowerPlus++;
      if(hit.ieta() < -29 && hit.energy() > hfTowerThreshold_) nHFtowerMinus++;

      if(useJets_){
	for(unsigned int j = 0 ; j < jets->size(); ++j){
	  const reco::Jet& jet = (*jets)[j];
	  double dr = reco::deltaR(myTowers.eta[myTowers.n],myTowers.phi[myTowers.n],jet.eta(),jet.phi());
	  if(dr < cone){ myTowers.isjet[myTowers.n] = true; }
	}
      }
      myTowers.n++;
    }

  }

  if(doBasicClusters_ && !doEbyEonly_){
    for(unsigned int j = 0 ; j < jets->size(); ++j){
      const reco::Jet& jet = (*jets)[j];
      myBC.n = 0;
      myBC.jtpt = jet.pt();
      myBC.jteta = jet.eta();
      myBC.jtphi = jet.phi();

      for(unsigned int i = 0; i < bClusters->size(); ++i){
	const reco::BasicCluster & bc= (*bClusters)[i];
	double dr = reco::deltaR(bc.eta(),bc.phi(),jet.eta(),jet.phi());
	if(dr < cone){
	  myBC.e[myBC.n] = bc.energy();
	  myBC.et[myBC.n] = bc.energy()*sin(bc.position().theta());
	  myBC.eta[myBC.n] = bc.eta();
	  myBC.phi[myBC.n] = bc.phi();
	  myBC.n++;
	}
      }
      bcTree->Fill();
    }
  }

  if(doCastor_){

    edm::Handle<CastorRecHitCollection> casrechits;
    try{ ev.getByLabel("castorreco",casrechits); }
    catch(...) { edm::LogWarning(" CASTOR ") << " Cannot get Castor RecHits " << std::endl; }

    int nhits = 0;
    double energyCastor = 0;

    if(casrechits.failedToGet()!=0 || !casrechits.isValid()) {
      edm::LogWarning(" CASTOR ") << " Cannot read CastorRecHitCollection" << std::endl;
    } else {
      for(size_t i1 = 0; i1 < casrechits->size(); ++i1) {
	const CastorRecHit & rh = (*casrechits)[i1];
	HcalCastorDetId castorid = rh.id();
	energyCastor += rh.energy();
	if (nhits  < 224) {
	  castorRecHit.e[nhits] = rh.energy();
	  castorRecHit.iphi[nhits] = castorid.sector();
	  castorRecHit.depth[nhits] = castorid.module();
	  castorRecHit.phi[nhits] = getPhi(castorid);
	  castorRecHit.saturation[nhits] = static_cast<int>( rh.flagField(HcalCaloFlagLabels::ADCSaturationBit) );

	}

	nhits++;

      } // end loop castor rechits
    }

    castorRecHit.n = nhits;
    castorTree->Fill();
  }

  if(doZDCRecHit_){

    edm::Handle<ZDCRecHitCollection> zdcrechits;

    try{ ev.getByLabel(zdcRecHitSrc_,zdcrechits); }
    catch(...) { edm::LogWarning(" ZDC ") << " Cannot get ZDC RecHits " << std::endl; }

    int nhits = 0;

    if (zdcrechits.failedToGet()!=0 || !zdcrechits.isValid()) {
      edm::LogWarning(" ZDC ") << " Cannot read ZDCRecHitCollection" << std::endl;
    } else {
      for(size_t i1 = 0; i1 < zdcrechits->size(); ++i1) {
	const ZDCRecHit & rh = (*zdcrechits)[i1];
	HcalZDCDetId zdcid = rh.id();
	if (nhits  < 18) {
	  zdcRecHit.e[nhits] = rh.energy();
	  zdcRecHit.zside[nhits] = zdcid.zside();
	  zdcRecHit.section[nhits] = zdcid.section();
	  zdcRecHit.channel[nhits] = zdcid.channel();
	  zdcRecHit.saturation[nhits] = static_cast<int>( rh.flagField(HcalCaloFlagLabels::ADCSaturationBit) );
	}

	nhits++;

      } // end loop zdc rechits
    }

    zdcRecHit.n = nhits;
    zdcRecHitTree->Fill();

  }

  if(doZDCDigi_){

    edm::Handle<ZDCDigiCollection> zdcdigis;

    try{ ev.getByLabel(zdcDigiSrc_,zdcdigis); }
    catch(...) { edm::LogWarning(" ZDC ") << " Cannot get ZDC Digis " << std::endl; }

    int nhits = 0;

    if (zdcdigis.failedToGet()!=0 || !zdcdigis.isValid()) {
      edm::LogWarning(" ZDC ") << " Cannot read ZDCDigiCollection" << std::endl;
    } else {

      edm::ESHandle<HcalDbService> conditions;
      iSetup.get<HcalDbRecord>().get(conditions);

      for(size_t i1 = 0; i1 < zdcdigis->size(); ++i1) {
	CaloSamples caldigi;
	const ZDCDataFrame & rh = (*zdcdigis)[i1];
	HcalZDCDetId zdcid = rh.id();

	if(calZDCDigi_){
 	  const HcalQIECoder* qiecoder=conditions->getHcalCoder(zdcid);
	  const HcalQIEShape* qieshape=conditions->getHcalShape(qiecoder);
	  HcalCoderDb coder(*qiecoder,*qieshape);
	  coder.adc2fC(rh,caldigi);
	}

	if (nhits  < 18) {
	  int ts = 0;
	  zdcDigi.zside[nhits] = zdcid.zside();
	  zdcDigi.section[nhits] = zdcid.section();
	  zdcDigi.channel[nhits] = zdcid.channel();

	  for(int j1 = 0; j1 < rh.size(); j1++){
	    zdcDigi.chargefC[ts][nhits]=calZDCDigi_?caldigi[ts]:rh[ts].nominal_fC();
	    zdcDigi.adc[ts][nhits]= rh[ts].adc();
	    ts++;
	  }
	}
	nhits++;
      } // end loop zdc rechits
    }

    zdcDigi.n = nhits;
    zdcDigiTree->Fill();

  }

  if(!doEbyEonly_){
    towerTree->Fill();

    if(doEcal_){
      eeTree->Fill();
      ebTree->Fill();
    }

    if(doHcal_){
      hbheTree->Fill();
      hfTree->Fill();
    }

    if (doFastJet_) {
      bkgTree->Fill();
    }
  }

  nt->Fill(nHFtowerPlus,nHFtowerMinus,nHFlongPlus,nHFlongMinus,nHFshortPlus,nHFshortMinus);

}


// ------------ method called once each job just before starting event loop  ------------
void
RecHitTreeProducer::beginJob()
{

  if(doHcal_){
    hbheTree = fs->make<TTree>("hbhe",versionTag);
    hbheTree->Branch("n",&hbheRecHit.n,"n/I");
    hbheTree->Branch("e",hbheRecHit.e,"e[n]/F");
    hbheTree->Branch("eraw",hbheRecHit.eraw,"eraw[n]/F");
    hbheTree->Branch("et",hbheRecHit.et,"et[n]/F");
    hbheTree->Branch("eta",hbheRecHit.eta,"eta[n]/F");
    hbheTree->Branch("phi",hbheRecHit.phi,"phi[n]/F");
    hbheTree->Branch("perp",hbheRecHit.perp,"perp[n]/F");

    hbheTree->Branch("isjet",hbheRecHit.isjet,"isjet[n]/O");
    hbheTree->Branch("depth",hfRecHit.depth,"depth[n]/I");

    hfTree = fs->make<TTree>("hf",versionTag);
    hfTree->Branch("n",&hfRecHit.n,"n/I");
    hfTree->Branch("e",hfRecHit.e,"e[n]/F");
    hfTree->Branch("et",hfRecHit.et,"et[n]/F");
    hfTree->Branch("eta",hfRecHit.eta,"eta[n]/F");
    hfTree->Branch("phi",hfRecHit.phi,"phi[n]/F");
    hfTree->Branch("perp",hfRecHit.perp,"perp[n]/F");
    hfTree->Branch("depth",hfRecHit.depth,"depth[n]/I");
    hfTree->Branch("isjet",hfRecHit.isjet,"isjet[n]/O");
  }

  if(doEcal_){

    eeTree = fs->make<TTree>("ee",versionTag);
    eeTree->Branch("n",&eeRecHit.n,"n/I");
    eeTree->Branch("e",eeRecHit.e,"e[n]/F");
    eeTree->Branch("et",eeRecHit.et,"et[n]/F");
    eeTree->Branch("eta",eeRecHit.eta,"eta[n]/F");
    eeTree->Branch("phi",eeRecHit.phi,"phi[n]/F");
    eeTree->Branch("perp",eeRecHit.perp,"perp[n]/F");
    eeTree->Branch("chi2",eeRecHit.chi2,"chi2[n]/F");
    eeTree->Branch("eError",eeRecHit.eError,"eError[n]/F");
    eeTree->Branch("flags",eeRecHit.flags,"flags[n]/i");

    eeTree->Branch("isjet",eeRecHit.isjet,"isjet[n]/O");

    ebTree = fs->make<TTree>("eb",versionTag);
    ebTree->Branch("n",&ebRecHit.n,"n/I");
    ebTree->Branch("e",ebRecHit.e,"e[n]/F");
    ebTree->Branch("et",ebRecHit.et,"et[n]/F");
    ebTree->Branch("eta",ebRecHit.eta,"eta[n]/F");
    ebTree->Branch("phi",ebRecHit.phi,"phi[n]/F");
    ebTree->Branch("perp",ebRecHit.perp,"perp[n]/F");
    ebTree->Branch("chi2",ebRecHit.chi2,"chi2[n]/F");
    ebTree->Branch("eError",ebRecHit.eError,"eError[n]/F");
    ebTree->Branch("flags",ebRecHit.flags,"flags[n]/i");
    
    ebTree->Branch("isjet",ebRecHit.isjet,"isjet[n]/O");
  }

  if(doTowers_){
    towerTree = fs->make<TTree>("tower",versionTag);
    towerTree->Branch("n",&myTowers.n,"n/I");
    towerTree->Branch("e",myTowers.e,"e[n]/F");
    towerTree->Branch("et",myTowers.et,"et[n]/F");
    towerTree->Branch("eta",myTowers.eta,"eta[n]/F");
    towerTree->Branch("phi",myTowers.phi,"phi[n]/F");
    towerTree->Branch("isjet",myTowers.isjet,"isjet[n]/O");
    towerTree->Branch("emEt",myTowers.emEt,"emEt[n]/F");
    towerTree->Branch("hadEt",myTowers.hadEt,"hadEt[n]/F");

    if(doVS_){

      towerTree->Branch("vsPt",myTowers.vsPt,"vsPt[n]/F");
      towerTree->Branch("vsPtInitial",myTowers.vsPtInitial,"vsPtInitial[n]/F");
      towerTree->Branch("vsArea",myTowers.vsArea,"vsArea[n]/F");

      towerTree->Branch("vn",myTowers.vn,Form("vn[%d][%d]/F",fourierOrder_,etaBins_));
      towerTree->Branch("psin",myTowers.psin,Form("vpsi[%d][%d]/F",fourierOrder_,etaBins_));
      towerTree->Branch("sumpt",myTowers.sumpt,Form("sumpt[%d]/F",etaBins_));
      if(doUEraw_){
	towerTree->Branch("ueraw",myTowers.ueraw,Form("ueraw[%d]/F",(fourierOrder_*etaBins_*2*3)));
      }
    }


  }


  if(doCastor_){
    castorTree = fs->make<TTree>("castor",versionTag);
    castorTree->Branch("n",&castorRecHit.n,"n/I");
    castorTree->Branch("e",castorRecHit.e,"e[n]/F");
    castorTree->Branch("iphi",castorRecHit.iphi,"iphi[n]/I");
    castorTree->Branch("phi",castorRecHit.phi,"phi[n]/F");
    castorTree->Branch("depth",castorRecHit.depth,"depth[n]/I");
    castorTree->Branch("saturation",castorRecHit.saturation,"saturation[n]/I");
  }

  if(doZDCRecHit_){
    zdcRecHitTree = fs->make<TTree>("zdcrechit",versionTag);
    zdcRecHitTree->Branch("n",&zdcRecHit.n,"n/I");
    zdcRecHitTree->Branch("e",zdcRecHit.e,"e[n]/F");
    zdcRecHitTree->Branch("saturation",zdcRecHit.saturation,"saturation[n]/F");
    zdcRecHitTree->Branch("zside",zdcRecHit.zside,"zside[n]/I");
    zdcRecHitTree->Branch("section",zdcRecHit.section,"section[n]/I");
    zdcRecHitTree->Branch("channel",zdcRecHit.channel,"channel[n]/I");
  }

  if(doZDCDigi_){
    TString nZdcTsSt="";
    nZdcTsSt+=nZdcTs_;

    zdcDigiTree = fs->make<TTree>("zdcdigi",versionTag);
    zdcDigiTree->Branch("n",&zdcDigi.n,"n/I");
    zdcDigiTree->Branch("zside",zdcDigi.zside,"zside[n]/I");
    zdcDigiTree->Branch("section",zdcDigi.section,"section[n]/I");
    zdcDigiTree->Branch("channel",zdcDigi.channel,"channel[n]/I");

    for( int i=0; i<nZdcTs_;i++){
      TString adcTsSt("adcTs"), chargefCTsSt("chargefCTs");
      adcTsSt+=i; chargefCTsSt+=i;

      zdcDigiTree->Branch(adcTsSt,zdcDigi.adc[i],adcTsSt+"[n]/I");
      zdcDigiTree->Branch(chargefCTsSt,zdcDigi.chargefC[i],chargefCTsSt+"[n]/F");
    }
  }

  if (saveBothVtx_) {
    towerTree->Branch("etVtx",myTowers.etVtx,"etvtx[n]/F");
    towerTree->Branch("etaVtx",myTowers.etaVtx,"etavtx[n]/F");
    towerTree->Branch("emEtVtx",myTowers.emEtVtx,"emEtVtx[n]/F");
    towerTree->Branch("hadEtVtx",myTowers.hadEtVtx,"hadEtVtx[n]/F");
  }

  if(doBasicClusters_){
    bcTree = fs->make<TTree>("bc",versionTag);
    bcTree->Branch("n",&myBC.n,"n/I");
    bcTree->Branch("e",myBC.e,"e[n]/F");
    bcTree->Branch("et",myBC.et,"et[n]/F");
    bcTree->Branch("eta",myBC.eta,"eta[n]/F");
    bcTree->Branch("phi",myBC.phi,"phi[n]/F");
    bcTree->Branch("jtpt",&myBC.jtpt,"jtpt/F");
    bcTree->Branch("jteta",&myBC.jteta,"jteta/F");
    bcTree->Branch("jtphi",&myBC.jtphi,"jtphi/F");
    //     bcTree->Branch("isjet",bcRecHit.isjet,"isjet[n]/O");
  }

  if(doFastJet_){
    bkgTree = fs->make<TTree>("bkg",versionTag);
    bkgTree->Branch("n",&bkg.n,"n/I");
    bkgTree->Branch("rho",bkg.rho,"rho[n]/F");
    bkgTree->Branch("sigma",bkg.sigma,"sigma[n]/F");
  }

  nt = fs->make<TNtuple>("ntEvent","","nHFplus:nHFminus:nHFlongPlus:nHFlongMinus:nHFshortPlus:nHFshortMinus");

}

// ------------ method called once each job just after ending the event loop  ------------
void
RecHitTreeProducer::endJob() {
}

math::XYZPoint RecHitTreeProducer::getPosition(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  math::XYZPoint posV(pos.x() - vtx.x(),pos.y() - vtx.y(),pos.z() - vtx.z());
  return posV;
}

double RecHitTreeProducer::getEt(math::XYZPoint pos, double energy){
  double et = energy*sin(pos.theta());
  return et;
}

double RecHitTreeProducer::getEt(const DetId &id, double energy, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = energy*sin(pos.theta());
  return et;
}

double RecHitTreeProducer::getEta(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = pos.eta();
  return et;
}

double RecHitTreeProducer::getPhi(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = pos.phi();
  return et;
}

double RecHitTreeProducer::getPerp(const DetId &id, reco::Vertex::Point vtx){
  const GlobalPoint& pos=geo->getPosition(id);
  double et = pos.perp();
  return et;
}

reco::Vertex::Point RecHitTreeProducer::getVtx(const edm::Event& ev)
{
  ev.getByLabel(VtxSrc_,vtxs);
  int greatestvtx = 0;
  int nVertex = vtxs->size();

  for (unsigned int i = 0 ; i< vtxs->size(); ++i){
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if( daughter > (*vtxs)[greatestvtx].tracksSize()) greatestvtx = i;
    //cout <<"Vertex: "<< (*vtxs)[i].position().z()<<" "<<daughter<<endl;
  }

  if(nVertex<=0){
    return reco::Vertex::Point(0,0,0);
  }
  return (*vtxs)[greatestvtx].position();
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitTreeProducer);
