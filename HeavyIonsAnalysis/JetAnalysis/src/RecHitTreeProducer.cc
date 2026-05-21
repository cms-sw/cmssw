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

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TTree.h"
#include "TNtuple.h"

#define MAXHITS 100000


struct MyRecHit{
  int n;

  uint32_t rawId[MAXHITS];
  int ieta[MAXHITS];
  int iphi[MAXHITS];
  // ix and iy are EE only
  int ix[MAXHITS];
  int iy[MAXHITS];
  int depth[MAXHITS];

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
  float emEtVtx[MAXHITS];
  float hadEtVtx[MAXHITS];

  int saturation[MAXHITS];

  float jtpt;
  float jteta;
  float jtphi;
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
  int    section[18];
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

class RecHitTreeProducer : public edm::one::EDAnalyzer<> {
public:
  explicit RecHitTreeProducer(const edm::ParameterSet&);
  ~RecHitTreeProducer();

  math::XYZPoint getPosition(const DetId &id, reco::Vertex::Point& vtx, const CaloGeometry& geo);
  double getEt(math::XYZPoint& pos, double energy, const CaloGeometry& geo);
  double getEt(const DetId &id, double energy, const CaloGeometry& geo);
  double getEta(const DetId &id, const CaloGeometry& geo);
  double getPhi(const DetId &id, const CaloGeometry& geo);
  double getPerp(const DetId &id, const CaloGeometry& geo);

  reco::Vertex::Point getVtx(const edm::Event& ev);

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::Handle<reco::CaloJetCollection> jets;

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
  //edm::ESHandle<CaloGeometry> geo;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;

  edm::EDGetTokenT<HFRecHitCollection> HcalRecHitHFSrc_;
  edm::EDGetTokenT<HBHERecHitCollection> HcalRecHitHBHESrc_;
  edm::EDGetTokenT<ZDCDigiCollection> zdcDigiSrc_;
  edm::EDGetTokenT<ZDCRecHitCollection> zdcRecHitSrc_;
  edm::EDGetTokenT<CastorRecHitCollection> castorRecHitSrc_;

  edm::EDGetTokenT<EcalRecHitCollection> EBSrc_;
  edm::EDGetTokenT<EcalRecHitCollection> EESrc_;
  edm::EDGetTokenT<reco::BasicClusterCollection> BCSrc_;
  edm::EDGetTokenT<CaloTowerCollection> TowerSrc_;

  edm::EDGetTokenT<reco::VertexCollection> VtxSrc_;
  edm::EDGetTokenT<reco::CaloJetCollection> JetSrc_;

  edm::EDGetTokenT<std::vector<double>> jetRhoSrc_;
  edm::EDGetTokenT<std::vector<double>> jetSigmaSrc_;

  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDatabaseToken_;

  bool useJets_;
  bool doBasicClusters_;
  bool doTowers_;
  bool doEcal_;
  bool doHBHE_;
  bool doHF_;
  bool doCastor_;
  bool doZDCRecHit_;
  bool doZDCDigi_;
  bool doFastJet_;

  bool hasVtx_;
  bool saveBothVtx_;

  bool doEbyEonly_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
constexpr double cone2 = 0.5 * 0.5;

//
// constructors and destructor
//
RecHitTreeProducer::RecHitTreeProducer(const edm::ParameterSet& iConfig):
  hcalDatabaseToken_(esConsumes<HcalDbService, HcalDbRecord>())
{
  //now do what ever initialization is needed
  doEbyEonly_ = iConfig.getParameter<bool>("doEbyEonly");

  doBasicClusters_ = iConfig.getParameter<bool>("doBasicClusters");
  doTowers_ = iConfig.getParameter<bool>("doTowers");
  doEcal_ = iConfig.getParameter<bool>("doEcal");
  doHBHE_ = iConfig.getParameter<bool>("doHBHE");
  doHF_ = iConfig.getParameter<bool>("doHF");
  doFastJet_ = iConfig.getParameter<bool>("doFastJet");
  doCastor_ = iConfig.getParameter<bool>("doCASTOR");
  doZDCRecHit_ = iConfig.getParameter<bool>("doZDCRecHit");
  doZDCDigi_ = iConfig.getParameter<bool>("doZDCDigi");

  useJets_ = iConfig.getParameter<bool>("useJets");

  if (!doEbyEonly_) {
    if (doEcal_) {
      EBSrc_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitSrc"));
      EESrc_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSrc"));
    }
    if (doHF_)
      HcalRecHitHFSrc_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hcalHFRecHitSrc"));
    if (doHBHE_)
      HcalRecHitHBHESrc_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hcalHBHERecHitSrc"));
    if (doBasicClusters_)
      BCSrc_ = consumes<reco::BasicClusterCollection>(iConfig.getParameter<edm::InputTag>("BasicClusterSrc"));
    if (doTowers_)
      TowerSrc_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("towersSrc"));

    if (useJets_)
      JetSrc_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("JetSrc"));
  }

  if (doFastJet_) {
    jetRhoSrc_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("FastJetRhoTag"));
    jetSigmaSrc_ = consumes<std::vector<double>>(iConfig.getParameter<edm::InputTag>("FastJetSigmaTag"));
  }

  if (doZDCDigi_)
    zdcDigiSrc_ = consumes<ZDCDigiCollection>(iConfig.getParameter<edm::InputTag>("zdcDigiSrc"));
  if (doZDCRecHit_)
    zdcRecHitSrc_ = consumes<ZDCRecHitCollection> (iConfig.getParameter<edm::InputTag>("zdcRecHitSrc"));
  if (doCastor_)
    castorRecHitSrc_ = consumes<CastorRecHitCollection> (iConfig.getParameter<edm::InputTag>("castorRecHitSrc"));

  hasVtx_ = iConfig.getParameter<bool>("hasVtx");
  if (hasVtx_)
    VtxSrc_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxSrc"));

  saveBothVtx_ = iConfig.getParameter<bool>("saveBothVtx");

  hfTowerThreshold_ = iConfig.getUntrackedParameter<double>("HFtowerMin",3.);
  hfLongThreshold_ = iConfig.getUntrackedParameter<double>("HFlongMin",0.5);
  hfShortThreshold_ = iConfig.getUntrackedParameter<double>("HFshortMin",0.85);
  hbhePtMin_ = iConfig.getUntrackedParameter<double>("HBHETreePtMin",0);
  hfPtMin_ = iConfig.getUntrackedParameter<double>("HFTreePtMin",0);
  ebPtMin_ = iConfig.getUntrackedParameter<double>("EBTreePtMin",0);
  eePtMin_ = iConfig.getUntrackedParameter<double>("EETreePtMin",0.);
  towerPtMin_ = iConfig.getUntrackedParameter<double>("TowerTreePtMin",0.);

  nZdcTs_ = iConfig.getParameter<int>("nZdcTs");
  calZDCDigi_ = iConfig.getParameter<bool>("calZDCDigi");

  geoToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
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

  //iSetup.get<CaloGeometryRecord>().get(geo);
  auto const& geo = iSetup.getData(geoToken_);

  // get vertex
  reco::Vertex::Point vtx(0,0,0);
  if (hasVtx_) vtx = getVtx(ev);
  if (useJets_ && !doEbyEonly_) ev.getByToken(JetSrc_,jets);

  if(doFastJet_){
    edm::Handle<std::vector<double>> rhos;
    ev.getByToken(jetRhoSrc_, rhos);
    edm::Handle<std::vector<double>> sigmas;
    ev.getByToken(jetSigmaSrc_, sigmas);

    bkg.n = rhos->size();
    for (unsigned int i = 0; i < rhos->size(); ++i) {
      bkg.rho[i] = (*rhos)[i];
      bkg.sigma[i] = (*sigmas)[i];
    }
    bkgTree->Fill();
  }

  int nHFlongPlus = 0;
  int nHFshortPlus = 0;
  int nHFtowerPlus = 0;
  int nHFlongMinus = 0;
  int nHFshortMinus = 0;
  int nHFtowerMinus = 0;

  if (doHF_ && !doEbyEonly_) {
    edm::Handle<HFRecHitCollection> hfHits;
    ev.getByToken(HcalRecHitHFSrc_, hfHits);

    for (auto const& hit : *hfHits) {
      if (getEt(hit.id(), hit.energy(), geo) < hfPtMin_) continue;

      const HcalDetId & id = hit.id();
      hfRecHit.rawId[hfRecHit.n] = id.rawId();
      hfRecHit.ieta[hfRecHit.n] = id.ieta();
      hfRecHit.iphi[hfRecHit.n] = id.iphi();

      hfRecHit.e[hfRecHit.n] = hit.energy();

      if(!saveBothVtx_){
        math::XYZPoint pos = getPosition(id,vtx, geo);
        hfRecHit.et[hfRecHit.n] = getEt(pos,hit.energy(), geo);
        hfRecHit.eta[hfRecHit.n] = pos.eta();
        hfRecHit.phi[hfRecHit.n] = pos.phi();
        hfRecHit.perp[hfRecHit.n] = pos.rho();
      }else{
        hfRecHit.et[hfRecHit.n] = getEt(id,hit.energy(), geo);
        hfRecHit.eta[hfRecHit.n] = getEta(id, geo);
        hfRecHit.phi[hfRecHit.n] = getPhi(id, geo);
        hfRecHit.perp[hfRecHit.n] = getPerp(id, geo);
      }

      hfRecHit.depth[hfRecHit.n] = id.depth();

      if(id.ieta() > 0){
        if(hit.energy() > hfShortThreshold_ && id.depth() != 1) nHFshortPlus++;
        if(hit.energy() > hfLongThreshold_ && id.depth() == 1) nHFlongPlus++;
      }else{
        if(hit.energy() > hfShortThreshold_ && id.depth() != 1) nHFshortMinus++;
        if(hit.energy() > hfLongThreshold_ && id.depth() == 1) nHFlongMinus++;
      }

      if(useJets_){
        for (auto const& jet : *jets) {
          double dr2 = reco::deltaR2(hfRecHit.eta[hfRecHit.n],hfRecHit.phi[hfRecHit.n],jet.eta(),jet.phi());
          hfRecHit.isjet[hfRecHit.n] = (dr2 < cone2);
        }
      }

      hfRecHit.n++;
    }
    hfTree->Fill();
  }

  if (doHBHE_ && !doEbyEonly_) {
    edm::Handle<HBHERecHitCollection> hbheHits;
    ev.getByToken(HcalRecHitHBHESrc_, hbheHits);

    for (auto const& hit : *hbheHits) {
      if (getEt(hit.id(), hit.energy(), geo) < hbhePtMin_) continue;

      const HcalDetId & id = hit.id();
      hbheRecHit.rawId[hbheRecHit.n] = id.rawId();
      hbheRecHit.ieta[hbheRecHit.n] = id.ieta();
      hbheRecHit.iphi[hbheRecHit.n] = id.iphi();

      hbheRecHit.e[hbheRecHit.n] = hit.energy();
      hbheRecHit.eraw[hbheRecHit.n] = hit.eraw();

      if(!saveBothVtx_){
        math::XYZPoint pos = getPosition(id,vtx, geo);
        hbheRecHit.et[hbheRecHit.n] = getEt(pos,hit.energy(), geo);
        hbheRecHit.eta[hbheRecHit.n] = pos.eta();
        hbheRecHit.phi[hbheRecHit.n] = pos.phi();
        hbheRecHit.perp[hbheRecHit.n] = pos.rho();
      }else{
        hbheRecHit.et[hbheRecHit.n] = getEt(id,hit.energy(), geo);
        hbheRecHit.eta[hbheRecHit.n] = getEta(id, geo);
        hbheRecHit.phi[hbheRecHit.n] = getPhi(id, geo);
        hbheRecHit.perp[hbheRecHit.n] = getPerp(id, geo);
      }

      hbheRecHit.depth[hbheRecHit.n] = id.depth();

      if(useJets_){
        for (auto const& jet : *jets) {
          double dr2 = reco::deltaR2(hbheRecHit.eta[hbheRecHit.n],hbheRecHit.phi[hbheRecHit.n],jet.eta(),jet.phi());
          hbheRecHit.isjet[hbheRecHit.n] = (dr2 < cone2);
        }
      }
      hbheRecHit.n++;
    }
    hbheTree->Fill();
  }

  if (doEcal_ && !doEbyEonly_) {
    edm::Handle<EcalRecHitCollection> ebHits;
    ev.getByToken(EBSrc_, ebHits);
    edm::Handle<EcalRecHitCollection> eeHits;
    ev.getByToken(EESrc_, eeHits);

    for(unsigned int i = 0; i < ebHits->size(); ++i){
      const EcalRecHit & hit= (*ebHits)[i];
      if (getEt(hit.id(),hit.energy(),geo)<ebPtMin_) continue;

      const DetId & id = hit.id();
      ebRecHit.rawId[ebRecHit.n] = id.rawId();
      ebRecHit.ieta[ebRecHit.n] = EBDetId(id).ieta();
      ebRecHit.iphi[ebRecHit.n] = EBDetId(id).iphi();

      ebRecHit.e[ebRecHit.n] = hit.energy();
      math::XYZPoint pos = getPosition(id,vtx, geo);

      if(!saveBothVtx_){
        ebRecHit.et[ebRecHit.n] = getEt(pos,hit.energy(), geo);
        ebRecHit.eta[ebRecHit.n] = pos.eta();
        ebRecHit.phi[ebRecHit.n] = pos.phi();
        ebRecHit.perp[ebRecHit.n] = pos.rho();
      }else{
        ebRecHit.et[ebRecHit.n] = getEt(id,hit.energy(), geo);
        ebRecHit.eta[ebRecHit.n] = getEta(id, geo);
        ebRecHit.phi[ebRecHit.n] = getPhi(id, geo);
        ebRecHit.perp[ebRecHit.n] = getPerp(id, geo);
      }
      ebRecHit.chi2[ebRecHit.n] = hit.chi2();
      ebRecHit.eError[ebRecHit.n] = hit.energyError();

      ebRecHit.flags[ebRecHit.n] = 0;
      for (uint32_t f=0; f<32; ++f) {
        if (hit.checkFlag(f))
          ebRecHit.flags[ebRecHit.n] |= 1 << f;
      }

      if(useJets_){
        for (auto const& jet : *jets) {
          double dr2 = reco::deltaR2(ebRecHit.eta[ebRecHit.n],ebRecHit.phi[ebRecHit.n],jet.eta(),jet.phi());
          ebRecHit.isjet[ebRecHit.n] = (dr2 < cone2);
        }
      }
      ebRecHit.n++;
    }
    ebTree->Fill();

    for(unsigned int i = 0; i < eeHits->size(); ++i){
      const EcalRecHit & hit= (*eeHits)[i];
      if (getEt(hit.id(),hit.energy(),geo)<eePtMin_) continue;

      const DetId &id = hit.id();
      eeRecHit.rawId[eeRecHit.n] = id.rawId();
      // ix and iy are EE only
      eeRecHit.ix[eeRecHit.n] = EEDetId(id).ix();
      eeRecHit.iy[eeRecHit.n] = EEDetId(id).iy()*EEDetId(id).zside();
      // positive (negative) eeRecHit.iy will correspond to EE+ (EE-)

      eeRecHit.e[eeRecHit.n] = hit.energy();
      math::XYZPoint pos = getPosition(id,vtx,geo);

      if(!saveBothVtx_){
        eeRecHit.et[eeRecHit.n] = getEt(pos,hit.energy(),geo);
        eeRecHit.eta[eeRecHit.n] = pos.eta();
        eeRecHit.phi[eeRecHit.n] = pos.phi();
        eeRecHit.perp[eeRecHit.n] = pos.rho();
      }else{
        eeRecHit.et[eeRecHit.n] = getEt(id,hit.energy(), geo);
        eeRecHit.eta[eeRecHit.n] = getEta(id, geo);
        eeRecHit.phi[eeRecHit.n] = getPhi(id,geo);
        eeRecHit.perp[eeRecHit.n] = getPerp(id, geo);
      }
      eeRecHit.chi2[eeRecHit.n] = hit.chi2();
      eeRecHit.eError[eeRecHit.n] = hit.energyError();

      eeRecHit.flags[eeRecHit.n] = 0;
      for (unsigned int f=0; f<32; ++f) {
        if (hit.checkFlag(f))
          eeRecHit.flags[eeRecHit.n] |= 1 << f;
      }

      if(useJets_){
        for (auto const& jet : *jets) {
          double dr2 = reco::deltaR2(eeRecHit.eta[eeRecHit.n],eeRecHit.phi[eeRecHit.n],jet.eta(),jet.phi());
          eeRecHit.isjet[eeRecHit.n] = (dr2 < cone2);
        }
      }
      eeRecHit.n++;
    }
    eeTree->Fill();
  }

  if (!doEbyEonly_ && doTowers_) {
    edm::Handle<CaloTowerCollection> towers;
    ev.getByToken(TowerSrc_, towers);

    for (auto const& hit : *towers) {
      if (hit.pt() < towerPtMin_) continue;

      const CaloTowerDetId & id = hit.id();
      myTowers.rawId[myTowers.n] = id.rawId();
      myTowers.ieta[myTowers.n] = id.ieta();
      myTowers.iphi[myTowers.n] = id.iphi();

      myTowers.e[myTowers.n] = hit.energy();
      myTowers.emEt[myTowers.n] = hit.emEt(vtx);
      myTowers.hadEt[myTowers.n] = hit.hadEt(vtx);

      if (!saveBothVtx_) {
        myTowers.et[myTowers.n] = hit.p4(vtx).Et();
        myTowers.eta[myTowers.n] = hit.p4(vtx).Eta();
        myTowers.phi[myTowers.n] = hit.p4(vtx).Phi();
      } else {
        myTowers.et[myTowers.n] = getEt(id,hit.energy(),geo);
        myTowers.eta[myTowers.n] = getEta(id,geo);
        myTowers.phi[myTowers.n] = getPhi(id,geo);

        myTowers.etVtx[myTowers.n] = hit.p4(vtx).Et();
        myTowers.etaVtx[myTowers.n] = hit.p4(vtx).Eta();
        myTowers.emEtVtx[myTowers.n] = hit.emEt(vtx);
        myTowers.hadEtVtx[myTowers.n] = hit.hadEt(vtx);
      }

      if(hit.ieta() > 29 && hit.energy() > hfTowerThreshold_) nHFtowerPlus++;
      if(hit.ieta() < -29 && hit.energy() > hfTowerThreshold_) nHFtowerMinus++;

      if(useJets_){
        for (auto const& jet : *jets) {
          double dr2 = reco::deltaR2(myTowers.eta[myTowers.n],myTowers.phi[myTowers.n],jet.eta(),jet.phi());
          myTowers.isjet[myTowers.n] = (dr2 < cone2);
        }
      }
      myTowers.n++;
    }

    towerTree->Fill();
  }

  if (doBasicClusters_ && !doEbyEonly_) {
    edm::Handle<reco::BasicClusterCollection> bClusters;
    ev.getByToken(BCSrc_, bClusters);

    for (auto const& jet : *jets) {
      myBC.n = 0;
      myBC.jtpt = jet.pt();
      myBC.jteta = jet.eta();
      myBC.jtphi = jet.phi();

      for (auto const& bc : *bClusters) {
        double dr2 = reco::deltaR2(bc.eta(),bc.phi(),jet.eta(),jet.phi());
        if (dr2 < cone2) {
          // rawId will be the rawId of the seed RecHit.
          const DetId & id = bc.hitsAndFractions().at(0).first;
          myBC.rawId[myBC.n] = id.rawId();
          if (id.subdetId() == EcalSubdetector::EcalBarrel) {
            myBC.ieta[myBC.n] = EBDetId(id).ieta();
            myBC.iphi[myBC.n] = EBDetId(id).iphi();
            myBC.ix[myBC.n] = -999;
            myBC.iy[myBC.n] = -999;
          } else if (id.subdetId() == EcalSubdetector::EcalEndcap) {
            myBC.ieta[myBC.n] = -999;
            myBC.iphi[myBC.n] = -999;
            myBC.ix[myBC.n] = EEDetId(id).ix();
            myBC.iy[myBC.n] = EEDetId(id).iy()*EEDetId(id).zside();
            // positive (negative) myBC.iy will correspond to EE+ (EE-)
          } else {
            myBC.ieta[myBC.n] = -999;
            myBC.iphi[myBC.n] = -999;
            myBC.ix[myBC.n] = -999;
            myBC.iy[myBC.n] = -999;
          }

          myBC.e[myBC.n] = bc.energy();
          myBC.et[myBC.n] =  bc.energy()*sin(bc.position().theta());
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
    ev.getByToken(castorRecHitSrc_, casrechits);

    int nhits = 0;
    for (auto const& rh : *casrechits) {
      HcalCastorDetId castorid = rh.id();
      if (nhits  < 224) {
        castorRecHit.rawId[nhits] = castorid.rawId();
        castorRecHit.e[nhits] = rh.energy();
        castorRecHit.iphi[nhits] = castorid.sector();
        castorRecHit.depth[nhits] = castorid.module();
        castorRecHit.phi[nhits] = getPhi(castorid,geo);
        castorRecHit.saturation[nhits] = static_cast<int>( rh.flagField(HcalCaloFlagLabels::ADCSaturationBit) );
      }

      nhits++;
    } // end loop castor rechits

    castorRecHit.n = nhits;
    castorTree->Fill();
  }

  if(doZDCRecHit_){
    edm::Handle<ZDCRecHitCollection> zdcrechits;
    ev.getByToken(zdcRecHitSrc_,zdcrechits);

    int nhits = 0;
    for (auto const& rh : *zdcrechits) {
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

    zdcRecHit.n = nhits;
    zdcRecHitTree->Fill();
  }

  if(doZDCDigi_){
    edm::Handle<ZDCDigiCollection> zdcdigis;
    ev.getByToken(zdcDigiSrc_,zdcdigis);

    edm::ESHandle<HcalDbService> conditions = iSetup.getHandle(hcalDatabaseToken_); 

    int nhits = 0;
    for (auto const& rh : *zdcdigis)  {
      HcalZDCDetId zdcid = rh.id();

      CaloSamples caldigi;
      if(calZDCDigi_){
        const HcalQIECoder* qiecoder = conditions->getHcalCoder(zdcid);
        const HcalQIEShape* qieshape = conditions->getHcalShape(qiecoder);
        HcalCoderDb coder(*qiecoder, *qieshape);
        coder.adc2fC(rh,caldigi);
      }

      if (nhits  < 18) {
        zdcDigi.zside[nhits] = zdcid.zside();
        zdcDigi.section[nhits] = zdcid.section();
        zdcDigi.channel[nhits] = zdcid.channel();

        for (int ts = 0; ts < rh.size(); ts++) {
          zdcDigi.chargefC[ts][nhits] = calZDCDigi_ ? caldigi[ts] : rh[ts].nominal_fC();
          zdcDigi.adc[ts][nhits] = rh[ts].adc();
        }
      }
      nhits++;
    } // end loop zdc rechits

    zdcDigi.n = nhits;
    zdcDigiTree->Fill();
  }

  if (!doEbyEonly_ && (doHF_ || doHBHE_))
    nt->Fill(nHFtowerPlus, nHFtowerMinus, nHFlongPlus, nHFlongMinus, nHFshortPlus, nHFshortMinus);
}


// ------------ method called once each job just before starting event loop  ------------
void
RecHitTreeProducer::beginJob()
{
  if (!doEbyEonly_) {
    if(doHBHE_){
      hbheTree = fs->make<TTree>("hbhe", "hbhe");
      hbheTree->Branch("n",&hbheRecHit.n,"n/I");
      hbheTree->Branch("e",hbheRecHit.e,"e[n]/F");
      hbheTree->Branch("eraw",hbheRecHit.eraw,"eraw[n]/F");
      hbheTree->Branch("et",hbheRecHit.et,"et[n]/F");
      hbheTree->Branch("eta",hbheRecHit.eta,"eta[n]/F");
      hbheTree->Branch("phi",hbheRecHit.phi,"phi[n]/F");
      hbheTree->Branch("perp",hbheRecHit.perp,"perp[n]/F");
      hbheTree->Branch("depth",hbheRecHit.depth,"depth[n]/I");
      hbheTree->Branch("rawId",hbheRecHit.rawId,"rawId[n]/I");
      hbheTree->Branch("ieta",hbheRecHit.ieta,"ieta[n]/I");
      hbheTree->Branch("iphi",hbheRecHit.iphi,"iphi[n]/I");
      if (useJets_)
        hbheTree->Branch("isjet",hbheRecHit.isjet,"isjet[n]/O");
    }

    if (doHF_) {
      hfTree = fs->make<TTree>("hf", "hf");
      hfTree->Branch("n",&hfRecHit.n,"n/I");
      hfTree->Branch("e",hfRecHit.e,"e[n]/F");
      hfTree->Branch("et",hfRecHit.et,"et[n]/F");
      hfTree->Branch("eta",hfRecHit.eta,"eta[n]/F");
      hfTree->Branch("phi",hfRecHit.phi,"phi[n]/F");
      hfTree->Branch("perp",hfRecHit.perp,"perp[n]/F");
      hfTree->Branch("depth",hfRecHit.depth,"depth[n]/I");
      hfTree->Branch("rawId",hfRecHit.rawId,"rawId[n]/I");
      hfTree->Branch("ieta",hfRecHit.ieta,"ieta[n]/I");
      hfTree->Branch("iphi",hfRecHit.iphi,"iphi[n]/I");
      if (useJets_)
        hfTree->Branch("isjet",hfRecHit.isjet,"isjet[n]/O");
    }

    if(doEcal_){
      eeTree = fs->make<TTree>("ee", "ee");
      eeTree->Branch("n",&eeRecHit.n,"n/I");
      eeTree->Branch("e",eeRecHit.e,"e[n]/F");
      eeTree->Branch("et",eeRecHit.et,"et[n]/F");
      eeTree->Branch("eta",eeRecHit.eta,"eta[n]/F");
      eeTree->Branch("phi",eeRecHit.phi,"phi[n]/F");
      eeTree->Branch("perp",eeRecHit.perp,"perp[n]/F");
      eeTree->Branch("rawId",eeRecHit.rawId,"rawId[n]/I");
      eeTree->Branch("ix",eeRecHit.ix,"ix[n]/I");
      eeTree->Branch("iy",eeRecHit.iy,"iy[n]/I");
      eeTree->Branch("chi2",eeRecHit.chi2,"chi2[n]/F");
      eeTree->Branch("eError",eeRecHit.eError,"eError[n]/F");
      eeTree->Branch("flags",eeRecHit.flags,"flags[n]/i");
      if (useJets_)
        eeTree->Branch("isjet",eeRecHit.isjet,"isjet[n]/O");

      ebTree = fs->make<TTree>("eb", "eb");
      ebTree->Branch("n",&ebRecHit.n,"n/I");
      ebTree->Branch("e",ebRecHit.e,"e[n]/F");
      ebTree->Branch("et",ebRecHit.et,"et[n]/F");
      ebTree->Branch("eta",ebRecHit.eta,"eta[n]/F");
      ebTree->Branch("phi",ebRecHit.phi,"phi[n]/F");
      ebTree->Branch("perp",ebRecHit.perp,"perp[n]/F");
      ebTree->Branch("rawId",ebRecHit.rawId,"rawId[n]/I");
      ebTree->Branch("ieta",ebRecHit.ieta,"ieta[n]/I");
      ebTree->Branch("iphi",ebRecHit.iphi,"iphi[n]/I");
      ebTree->Branch("chi2",ebRecHit.chi2,"chi2[n]/F");
      ebTree->Branch("eError",ebRecHit.eError,"eError[n]/F");
      ebTree->Branch("flags",ebRecHit.flags,"flags[n]/i");
      if (useJets_)
        ebTree->Branch("isjet",ebRecHit.isjet,"isjet[n]/O");
    }

    if(doTowers_){
      towerTree = fs->make<TTree>("tower", "tower");
      towerTree->Branch("n",&myTowers.n,"n/I");
      towerTree->Branch("e",myTowers.e,"e[n]/F");
      towerTree->Branch("et",myTowers.et,"et[n]/F");
      towerTree->Branch("eta",myTowers.eta,"eta[n]/F");
      towerTree->Branch("phi",myTowers.phi,"phi[n]/F");
      towerTree->Branch("emEt",myTowers.emEt,"emEt[n]/F");
      towerTree->Branch("hadEt",myTowers.hadEt,"hadEt[n]/F");
      towerTree->Branch("rawId",myTowers.rawId,"rawId[n]/I");
      towerTree->Branch("ieta",myTowers.ieta,"ieta[n]/I");
      towerTree->Branch("iphi",myTowers.iphi,"iphi[n]/I");
      if (useJets_)
        towerTree->Branch("isjet",myTowers.isjet,"isjet[n]/O");

      if (saveBothVtx_) {
        towerTree->Branch("etVtx",myTowers.etVtx,"etvtx[n]/F");
        towerTree->Branch("etaVtx",myTowers.etaVtx,"etavtx[n]/F");
        towerTree->Branch("emEtVtx",myTowers.emEtVtx,"emEtVtx[n]/F");
        towerTree->Branch("hadEtVtx",myTowers.hadEtVtx,"hadEtVtx[n]/F");
      }
    }

    if(doBasicClusters_){
      bcTree = fs->make<TTree>("bc", "clusters");
      bcTree->Branch("n",&myBC.n,"n/I");
      bcTree->Branch("e",myBC.e,"e[n]/F");
      bcTree->Branch("et",myBC.et,"et[n]/F");
      bcTree->Branch("eta",myBC.eta,"eta[n]/F");
      bcTree->Branch("phi",myBC.phi,"phi[n]/F");
      bcTree->Branch("jtpt",&myBC.jtpt,"jtpt/F");
      bcTree->Branch("jteta",&myBC.jteta,"jteta/F");
      bcTree->Branch("jtphi",&myBC.jtphi,"jtphi/F");
      bcTree->Branch("rawId",myBC.rawId,"rawId[n]/I");
      bcTree->Branch("ieta",myBC.ieta,"ieta[n]/I");
      bcTree->Branch("iphi",myBC.iphi,"iphi[n]/I");
      bcTree->Branch("ix",myBC.ix,"ix[n]/I");
      bcTree->Branch("iy",myBC.iy,"iy[n]/I");
    }

    if (doTowers_ || doHF_)
      nt = fs->make<TNtuple>("ntEvent","","nHFplus:nHFminus:nHFlongPlus:nHFlongMinus:nHFshortPlus:nHFshortMinus");
  }

  if(doCastor_){
    castorTree = fs->make<TTree>("castor", "castor");
    castorTree->Branch("n",&castorRecHit.n,"n/I");
    castorTree->Branch("e",castorRecHit.e,"e[n]/F");
    castorTree->Branch("iphi",castorRecHit.iphi,"iphi[n]/I");
    castorTree->Branch("phi",castorRecHit.phi,"phi[n]/F");
    castorTree->Branch("depth",castorRecHit.depth,"depth[n]/I");
    castorTree->Branch("saturation",castorRecHit.saturation,"saturation[n]/I");
    castorTree->Branch("rawId",castorRecHit.rawId,"rawId[n]/I");
  }

  if(doZDCRecHit_){
    zdcRecHitTree = fs->make<TTree>("zdcrechit", "zdc");
    zdcRecHitTree->Branch("n",&zdcRecHit.n,"n/I");
    zdcRecHitTree->Branch("e",zdcRecHit.e,"e[n]/F");
    zdcRecHitTree->Branch("saturation",zdcRecHit.saturation,"saturation[n]/F");
    zdcRecHitTree->Branch("zside",zdcRecHit.zside,"zside[n]/I");
    zdcRecHitTree->Branch("section",zdcRecHit.section,"section[n]/I");
    zdcRecHitTree->Branch("channel",zdcRecHit.channel,"channel[n]/I");
  }

  if(doZDCDigi_){
    zdcDigiTree = fs->make<TTree>("zdcdigi", "zdc");
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

  if(doFastJet_){
    bkgTree = fs->make<TTree>("bkg", "bkg");
    bkgTree->Branch("n",&bkg.n,"n/I");
    bkgTree->Branch("rho",bkg.rho,"rho[n]/F");
    bkgTree->Branch("sigma",bkg.sigma,"sigma[n]/F");
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void
RecHitTreeProducer::endJob() {
}


math::XYZPoint RecHitTreeProducer::getPosition(const DetId &id, reco::Vertex::Point& vtx, const CaloGeometry& geo) {
  const GlobalPoint& pos = geo.getPosition(id);
  return math::XYZPoint(pos.x() - vtx.x(), pos.y() - vtx.y(), pos.z() - vtx.z());
}

inline double RecHitTreeProducer::getEt(math::XYZPoint& pos, double energy, const CaloGeometry& geo) {
  return energy * sin(pos.theta());
}

inline double RecHitTreeProducer::getEt(const DetId &id, double energy, const CaloGeometry& geo) {
  const GlobalPoint& pos = geo.getPosition(id);
  return energy * sin(pos.theta());
}

inline double RecHitTreeProducer::getEta(const DetId &id, const CaloGeometry& geo) {
  const GlobalPoint& pos = geo.getPosition(id);
  return pos.eta();
}

inline double RecHitTreeProducer::getPhi(const DetId &id, const CaloGeometry& geo) {
  const GlobalPoint& pos = geo.getPosition(id);
  return pos.phi();
}

inline double RecHitTreeProducer::getPerp(const DetId &id, const CaloGeometry& geo) {
  const GlobalPoint& pos = geo.getPosition(id);
  return pos.perp();
}

reco::Vertex::Point RecHitTreeProducer::getVtx(const edm::Event& ev) {
  edm::Handle<reco::VertexCollection> vtxs;
  ev.getByToken(VtxSrc_, vtxs);

  if (vtxs->empty())
    return reco::Vertex::Point(0,0,0);

  int greatestvtx = 0;
  for (unsigned int i = 0; i < vtxs->size(); ++i) {
    unsigned int daughter = (*vtxs)[i].tracksSize();
    if (daughter > (*vtxs)[greatestvtx].tracksSize())
      greatestvtx = i;
  }

  return (*vtxs)[greatestvtx].position();
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitTreeProducer);
