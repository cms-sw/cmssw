// -*- C++ -*-
//
// Package:    EvtPlaneProducer
// Class:      EvtPlaneProducer
// 
/**\class EvtPlaneProducer EvtPlaneProducer.cc RecoHI/EvtPlaneProducer/src/EvtPlaneProducer.cc
   
Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Sergey Petrushanko
//         Created:  Fri Jul 11 10:05:00 2008
// $Id: EvtPlaneProducer.cc,v 1.18 2011/10/07 09:41:29 yilmaz Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <time.h>
#include "TMath.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include <cstdlib>
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"
#include "CondFormats/DataRecord/interface/HeavyIonRPRcd.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlatten.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/LoadEPDB.h"

using namespace std;
using namespace hi;

//
// class decleration
//

class EvtPlaneProducer : public edm::EDProducer {
public:
  explicit EvtPlaneProducer(const edm::ParameterSet&);
  ~EvtPlaneProducer();
  
private:
  //edm::Service<TFileService> fs;
  class GenPlane {
  public: 
    GenPlane(string name,double etaminval1,double etamaxval1,double etaminval2,double etamaxval2,int orderval){
      epname=name;
      etamin1=etaminval1;
      etamax1=etamaxval1;
      etamin2=etaminval2;
      etamax2=etamaxval2;
      sumsin=0;
      sumcos=0;
      sumsinNoWgt=0;
      sumcosNoWgt=0;

      mult = 0;
      order = (double) orderval;
    }
    ~GenPlane(){;}
    void addParticle(double w, double PtOrEt, double s, double c, double eta) {
      if((eta>=etamin1 && eta<etamax1) || 
	 (etamin2!= etamax2 && eta>=etamin2 && eta<etamax2 )) {
	sumsin+=w*s;
	sumcos+=w*c;
	sumsinNoWgt+=s;
	sumcosNoWgt+=c;

	sumw+=w;
	sumw2+=w*w;
	sumPtOrEt+=PtOrEt;
	sumPtOrEt2+=PtOrEt*PtOrEt;
	++mult;
      }
    }
    
    double getAngle(double &ang, double &sv, double &cv, double &svNoWgt, double &cvNoWgt,  double &w, double &w2, double &PtOrEt, double &PtOrEt2, uint &epmult){
      ang = -10;
      sv = 0;
      cv = 0;
      sv = sumsin;
      cv = sumcos;
      svNoWgt = sumsinNoWgt;
      cvNoWgt = sumcosNoWgt;
      w = sumw;
      w2 = sumw2;
      PtOrEt = sumPtOrEt;
      PtOrEt2 = sumPtOrEt2;
      epmult = mult;
      double q = sv*sv+cv*cv;
      if(q>0) ang = atan2(sv,cv)/order;
      return ang;
    }
    void reset() {
      sumsin=0;
      sumcos=0;
      sumsinNoWgt = 0;
      sumcosNoWgt = 0;
      sumw = 0;
      sumw2 = 0;
      mult = 0;
      sumPtOrEt = 0;
      sumPtOrEt2 = 0;
    }
  private:
    string epname;
    double etamin1;
    double etamax1;

    double etamin2;
    double etamax2;
    double sumsin;
    double sumcos;
    double sumsinNoWgt;
    double sumcosNoWgt;
    uint mult;
    double sumw;
    double sumw2;
    double sumPtOrEt;
    double sumPtOrEt2;
    double order;
  };
  

  GenPlane *rp[NumEPNames];

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  // edm::InputTag vtxCollection_;
  // edm::InputTag caloCollection_;
  // edm::InputTag trackCollection_;

  edm::Service<TFileService> fs;

  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;

  edm::InputTag centralityBinTag_;
  edm::EDGetTokenT<int> centralityBinToken;
  edm::Handle<int> cbin_;


  edm::InputTag vertexTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertexToken;
  edm::Handle<std::vector<reco::Vertex>> vertex_;

  edm::InputTag caloTag_;
  edm::EDGetTokenT<CaloTowerCollection> caloToken;
  edm::Handle<CaloTowerCollection> caloCollection_;

  edm::InputTag castorTag_;
  edm::EDGetTokenT<std::vector<reco::CastorTower>> castorToken;
  edm::Handle<std::vector<reco::CastorTower>> castorCollection_;

  edm::InputTag trackTag_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken;
  edm::Handle<reco::TrackCollection> trackCollection_;

  bool foundCentTag;
  bool useECAL_;
  bool useHCAL_;
  bool useTrack_;
  bool loadDB_;
  double minet_;
  double maxet_;
  double effm_;
  double minpt_;
  double maxpt_;
  double minvtx_;
  double maxvtx_;
  double dzerr_;
  double chi2_;
  bool FirstEvent;
  int FlatOrder_;
  uint runno_; 
  int NumFlatBins_;
  double nCentBins_;
  double caloCentRef_;
  double caloCentRefWidth_;
  int caloCentRefMinBin_;
  int caloCentRefMaxBin_;
  int CentBinCompression_;
  HiEvtPlaneFlatten * flat[NumEPNames];
};

EvtPlaneProducer::EvtPlaneProducer(const edm::ParameterSet& iConfig) {

  nCentBins_ = 200.;

  centralityVariable_ = iConfig.getParameter<std::string>("centralityVariable");
  if(iConfig.exists("nonDefaultGlauberModel")){
    centralityMC_ = iConfig.getParameter<std::string>("nonDefaultGlauberModel");
  }
  centralityLabel_ = centralityVariable_+centralityMC_;

  centralityBinTag_ = iConfig.getParameter<edm::InputTag>("centralityBinTag_");
  centralityBinToken = consumes<int>(centralityBinTag_);

  vertexTag_  = iConfig.getParameter<edm::InputTag>("vertexTag_");
  vertexToken = consumes<std::vector<reco::Vertex>>(vertexTag_);

  caloTag_ = iConfig.getParameter<edm::InputTag>("caloTag_");
  caloToken = consumes<CaloTowerCollection>(caloTag_);

  castorTag_ = iConfig.getParameter<edm::InputTag>("castorTag_");
  castorToken = consumes<std::vector<reco::CastorTower>>(castorTag_);

  trackTag_ = iConfig.getParameter<edm::InputTag>("trackTag_");
  trackToken = consumes<reco::TrackCollection>(trackTag_);

  FlatOrder_ = iConfig.getUntrackedParameter<int>("FlatOrder_", 9);
  NumFlatBins_ = iConfig.getUntrackedParameter<int>("NumFlatBins_",20);
  CentBinCompression_ = iConfig.getUntrackedParameter<int>("CentBinCompression_",5);
  caloCentRef_ = iConfig.getUntrackedParameter<double>("caloCentRef_",80.);
  caloCentRefWidth_ = iConfig.getUntrackedParameter<double>("caloCentRefWidth_",5.);
  loadDB_ = iConfig.getUntrackedParameter<bool>("loadDB_",true);

  minet_ = iConfig.getUntrackedParameter<double>("minet_",-1.);
  maxet_ = iConfig.getUntrackedParameter<double>("maxet_",-1.);
  minpt_ = iConfig.getUntrackedParameter<double>("minpt_",0.3);
  maxpt_ = iConfig.getUntrackedParameter<double>("maxpt_",3.0);
  minvtx_ = iConfig.getUntrackedParameter<double>("minvtx_",-25.);
  maxvtx_ = iConfig.getUntrackedParameter<double>("maxvtx_",25.);
  dzerr_ = iConfig.getUntrackedParameter<double>("dzerr_",10.);
  chi2_  = iConfig.getUntrackedParameter<double>("chi2_",40.);
  FirstEvent = kTRUE;
  produces<reco::EvtPlaneCollection>("recoLevel");
  for(int i = 0; i<NumEPNames; i++ ) {
    rp[i] = new GenPlane(EPNames[i].data(),EPEtaMin1[i],EPEtaMax1[i],EPEtaMin2[i],EPEtaMax2[i],EPOrder[i]);
  }
  for(int i = 0; i<NumEPNames; i++) {
    flat[i] = new HiEvtPlaneFlatten();
    flat[i]->Init(FlatOrder_,NumFlatBins_,EPNames[i],EPOrder[i]);
  }

}


EvtPlaneProducer::~EvtPlaneProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EvtPlaneProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  using namespace reco;
  
  bool newrun = false;
  if(runno_ != iEvent.id().run()) newrun = true;
  runno_ = iEvent.id().run();

  if( (FirstEvent && loadDB_)|| (newrun&&loadDB_)) {
    FirstEvent = kFALSE;
    newrun = false;
    //
    //Get Size of Centrality Table
    //
    edm::ESHandle<CentralityTable> centDB_;
    iSetup.get<HeavyIonRcd>().get(centralityLabel_,centDB_);
    nCentBins_ = centDB_->m_table.size();
    for(int i = 0; i<NumEPNames; i++) {
      flat[i]->SetCaloCentRefBins(-1,-1);
      if(caloCentRef_>0) {
	int minbin = (caloCentRef_-caloCentRefWidth_/2.)*nCentBins_/100.;
	int maxbin = (caloCentRef_+caloCentRefWidth_/2.)*nCentBins_/100.;
	minbin/=CentBinCompression_;
	maxbin/=CentBinCompression_;
	if(minbin>0 && maxbin>=minbin) {
	  if(EPDet[i]==HF || EPDet[i]==Castor) flat[i]->SetCaloCentRefBins(minbin,maxbin);
	}
      }
    }


    //
    //Get flattening parameter file.  
    //
    edm::ESHandle<RPFlatParams> flatparmsDB_;
    iSetup.get<HeavyIonRPRcd>().get(flatparmsDB_);
    LoadEPDB * db = new LoadEPDB(flatparmsDB_,flat);
    if(!db->IsSuccess()) {
      loadDB_ = kFALSE;
    }
    
    
  } //First event
  
  //
  //Get Centrality
  //
  int bin = 0;
  iEvent.getByToken(centralityBinToken, cbin_);
  int cbin = *cbin_;
  bin = cbin/CentBinCompression_; 

  //
  //Get Vertex
  //
  int vs_sell;
  float vzr_sell;
  iEvent.getByToken(vertexToken,vertex_);
  const reco::VertexCollection * vertices3 = vertex_.product();
  vs_sell = vertices3->size();
  if(vs_sell>0) {
    vzr_sell = vertices3->begin()->z();
  } else
    vzr_sell = -999.9;
  //
  for(int i = 0; i<NumEPNames; i++) rp[i]->reset();
  if(vzr_sell>minvtx_ && vzr_sell<maxvtx_) {
  
    //calorimetry part
    
    double tower_eta, tower_phi;
    double tower_energyet, tower_energyet_e, tower_energyet_h;
 
    iEvent.getByToken(caloToken,caloCollection_);
    
    if(caloCollection_.isValid()){
      for (CaloTowerCollection::const_iterator j = caloCollection_->begin();j !=caloCollection_->end(); j++) {   
	tower_eta        = j->eta();
	tower_phi        = j->phi();
	tower_energyet_e   = j->emEt();
	tower_energyet_h   = j->hadEt();
	tower_energyet     = tower_energyet_e + tower_energyet_h;
	double minet = minet_;
	double maxet = maxet_;
	for(int i = 0; i<NumEPNames; i++) {
	  if(minet_<0) minet = minTransverse[i];
	  if(maxet_<0) maxet = maxTransverse[i];
	  if(tower_energyet<minet) continue;
	  if(tower_energyet>maxet) continue;
	  if(EPDet[i]==HF) {
	    double w = tower_energyet*flat[i]->GetEtScale(vzr_sell,bin);
	    if(EPOrder[i]==1 ) {
	      if(MomConsWeight[i][0]=='y' && loadDB_ ) {
		w = flat[i]->GetW(tower_energyet, vzr_sell, bin);
	      }
	      if(tower_eta<0 ) w=-w;
	    }
	    rp[i]->addParticle(w,tower_energyet,sin(EPOrder[i]*tower_phi),cos(EPOrder[i]*tower_phi),tower_eta);
	  }
	}
      } 
    }

    //Castor part
    
 
    iEvent.getByToken(castorToken,castorCollection_);
    
    if(castorCollection_.isValid()){
      for (std::vector<reco::CastorTower>::const_iterator j = castorCollection_->begin();j !=castorCollection_->end(); j++) {   
       	tower_eta        = j->position().eta();
	double labang = 2*TMath::ATan(TMath::Exp(-tower_eta));
       	tower_phi        = j->position().phi();
       	tower_energyet     = j->energy()*TMath::Sin(labang);
	double minet = minet_;
	double maxet = maxet_;
       	for(int i = 0; i<NumEPNames; i++) {
       	  if(EPDet[i]==Castor) {
	    if(minet_<0) minet = minTransverse[i];
	    if(maxet_<0) maxet = maxTransverse[i];
	    if(tower_energyet<minet) continue;
	    if(tower_energyet>maxet) continue;
       	    double w = tower_energyet;
       	    if(EPOrder[i]==1 ) {
       	      if(MomConsWeight[i][0]=='y' && loadDB_ ) {
       		w = flat[i]->GetW(tower_energyet, vzr_sell, bin);
       	      }
       	      if(tower_eta<0 ) w=-w;
       	    }
       	    rp[i]->addParticle(w,tower_energyet,sin(EPOrder[i]*tower_phi),cos(EPOrder[i]*tower_phi),tower_eta);
       	  }
       	}
      } 
    }

    //Tracking part
    
    double track_eta;
    double track_phi;
    double track_pt;
   
    iEvent.getByToken(trackToken, trackCollection_);
    if(trackCollection_.isValid()){
      for(reco::TrackCollection::const_iterator j = trackCollection_->begin(); j != trackCollection_->end(); j++){	
	iEvent.getByLabel(vertexTag_, vertex_);	    
	math::XYZPoint vtxPoint(0.0,0.0,0.0);
	double vzErr =0.0, vxErr=0.0, vyErr=0.0;
	if(vertex_->size()>0) {
	  vtxPoint=vertex_->begin()->position();
	  vzErr=vertex_->begin()->zError();
	  vxErr=vertex_->begin()->xError();
	  vyErr=vertex_->begin()->yError();
	}
	bool accepted = true;
	bool isPixel = false;
	// determine if the track is a pixel track
	if ( j->numberOfValidHits() < 7 ) isPixel = true;
	
	// determine the vertex significance 
	double d0=0.0, dz=0.0, d0sigma=0.0, dzsigma=0.0;
	d0 = -1.*j->dxy(vtxPoint);
	dz = j->dz(vtxPoint);
	d0sigma = sqrt(j->d0Error()*j->d0Error()+vxErr*vyErr);
	dzsigma = sqrt(j->dzError()*j->dzError()+vzErr*vzErr);
	
	// cuts for pixel tracks
	if( isPixel ){
	  // dz significance cut 
	  if ( fabs(dz/dzsigma) > dzerr_ ) accepted = false;
	  // chi2/ndof cut 
	  if ( j->normalizedChi2() > chi2_ ) accepted = false;
	}   
	// cuts for full tracks
	if ( ! isPixel) {
	  // dz and d0 significance cuts 
	  if ( fabs(dz/dzsigma) > 3 ) accepted = false;
	  if ( fabs(d0/d0sigma) > 3 ) accepted = false;
	  // pt resolution cut
	  if ( j->ptError()/j->pt() > 0.1 ) accepted = false;
	  // number of valid hits cut
	  if ( j->numberOfValidHits() < 12 ) accepted = false;
	}
	if( accepted ) {
	  track_eta = j->eta();
	  track_phi = j->phi();
	  track_pt = j->pt();
	  double minpt = minpt_;
	  double maxpt = maxpt_;
	  for(int i = 0; i<NumEPNames; i++) {
	    if(minpt_<0) minpt = minTransverse[i];
	    if(maxpt_<0) maxpt = maxTransverse[i];
	    if(track_pt<minpt) continue;
	    if(track_pt>maxpt) continue;
	    if(EPDet[i]==Tracker) {
	      double w = track_pt;
	      if(w>2.5) w=2.0;   //v2 starts decreasing above ~2.5 GeV/c
	      if(EPOrder[i]==1) {
		if(MomConsWeight[i][0]=='y' && loadDB_) {
		  w = flat[i]->GetW(track_pt, vzr_sell, bin);
		}
		if(track_eta<0) w=-w;
	      }
	      rp[i]->addParticle(w,track_pt,sin(EPOrder[i]*track_phi),cos(EPOrder[i]*track_phi),track_eta);
	    }
	  }
	}  
      } //end for
    }
    
    std::auto_ptr<EvtPlaneCollection> evtplaneOutput(new EvtPlaneCollection);
    EvtPlane *ep[NumEPNames];
    
    double ang=-10;
    double sv = 0;
    double cv = 0;
    double svNoWgt = 0;
    double cvNoWgt = 0;

    double wv = 0;
    double wv2 = 0;
    double pe = 0;
    double pe2 = 0;
    uint epmult = 0;

    for(int i = 0; i<NumEPNames; i++) {
      rp[i]->getAngle(ang,sv,cv,svNoWgt, cvNoWgt, wv,wv2,pe,pe2,epmult);
      ep[i] = new EvtPlane(i,0,ang,sv,cv,wv,wv2,pe,pe2,epmult);
      ep[i]->AddLevel(3, 0., svNoWgt, cvNoWgt);
    }

    for(int i = 0; i<NumEPNames; i++) {
      evtplaneOutput->push_back(*ep[i]);
    }  
    
    
    iEvent.put(evtplaneOutput, "recoLevel");
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
EvtPlaneProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EvtPlaneProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EvtPlaneProducer);
