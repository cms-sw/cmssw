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
// $Id: EvtPlaneProducer.cc,v 1.21 2012/02/15 11:04:09 eulisse Exp $
//
//
#define TRACKCOLLECTION 1
//#define RECOCHARGEDCANDIDATECOLLECTION 1


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
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <cstdlib>
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

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
      order = (double) orderval;
    }
    ~GenPlane(){;}
    void addParticle(double w, double s, double c, double eta) {
      if(w < 0.001) return;
      if((eta>=etamin1 && eta<etamax1) || 
	 (etamin2!= etamax2 && eta>=etamin2 && eta<etamax2 )) {
	sumsin+=w*s;
	sumcos+=w*c;
	//For tracking, w=1 and mult is the track multiplicity.  
	//For calorimetors, w is an energy that needs to be subsequently 
	//converted if an  multiplicity if needed
	mult+=w;
      }
    }
    
    double getAngle(double &ang, double &sv, double &cv){
      ang = -10;
      sv = 0;
      cv = 0;
      sv = sumsin;
      cv = sumcos;
      double q = sv*sv+cv*cv;
      if(q>0) ang = atan2(sv,cv)/order;
      return ang;
    }
    void reset() {
      sumsin=0;
      sumcos=0;
      mult = 0;
    }
  private:
    string epname;
    double etamin1;
    double etamax1;

    double etamin2;
    double etamax2;
    double sumsin;
    double sumcos;
    int mult;
    double order;
  };
  

  GenPlane *rp[NumEPNames];

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  edm::InputTag vtxCollection_;
  edm::InputTag caloCollection_;
  edm::InputTag trackCollection_;
  bool useECAL_;
  bool useHCAL_;
  bool useTrack_;
  bool useTrackPtWeight_;
  double minet_;
  double maxet_;
  double minpt_;
  double maxpt_;
  double minvtx_;
  double maxvtx_;
  double dzerr_;
  double chi2_;

  bool storeNames_;
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
EvtPlaneProducer::EvtPlaneProducer(const edm::ParameterSet& iConfig)
{
  
  
  //register your products
  vtxCollection_  = iConfig.getParameter<edm::InputTag>("vtxCollection_");
  caloCollection_  = iConfig.getParameter<edm::InputTag>("caloCollection_");
  trackCollection_  = iConfig.getParameter<edm::InputTag>("trackCollection_");
  useECAL_ = iConfig.getUntrackedParameter<bool>("useECAL_",true);
  useHCAL_ = iConfig.getUntrackedParameter<bool>("useHCAL_",true);
  useTrack_ = iConfig.getUntrackedParameter<bool>("useTrack",true);
  useTrackPtWeight_ = iConfig.getUntrackedParameter<bool>("useTrackPtWeight_",true);
  minet_ = iConfig.getUntrackedParameter<double>("minet_",0.2);
  maxet_ = iConfig.getUntrackedParameter<double>("maxet_",500.);
  minpt_ = iConfig.getUntrackedParameter<double>("minpt_",0.3);
  maxpt_ = iConfig.getUntrackedParameter<double>("maxpt_",2.5);
  minvtx_ = iConfig.getUntrackedParameter<double>("minvtx_",-50.);
  maxvtx_ = iConfig.getUntrackedParameter<double>("maxvtx_",50.);
  dzerr_ = iConfig.getUntrackedParameter<double>("dzerr_",10.);
  chi2_  = iConfig.getUntrackedParameter<double>("chi2_",40.);

  storeNames_ = 1;
  produces<reco::EvtPlaneCollection>("recoLevel");
  for(int i = 0; i<NumEPNames; i++ ) {
    rp[i] = new GenPlane(EPNames[i].data(),EPEtaMin1[i],EPEtaMax1[i],EPEtaMin2[i],EPEtaMax2[i],EPOrder[i]);
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
  using namespace reco;

  int vs_sell;
  float vzr_sell;
  //
  //Get Vertex
  //
  edm::Handle<reco::VertexCollection> vertexCollection3;
  iEvent.getByLabel(vtxCollection_,vertexCollection3);
  const reco::VertexCollection * vertices3 = vertexCollection3.product();
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
    double s1, s2, s13,s23,s14,s24,s15,s25,s16,s26;
    Handle<CaloTowerCollection> calotower;
    iEvent.getByLabel(caloCollection_,calotower);
    
    if(calotower.isValid()){
      
      for (CaloTowerCollection::const_iterator j = calotower->begin();j !=calotower->end(); j++) {   
	tower_eta        = j->eta();
	tower_phi        = j->phi();
	tower_energyet_e   = j->emEt();
	tower_energyet_h   = j->hadEt();
	tower_energyet     = tower_energyet_e + tower_energyet_h;
	
	s1 = sin(2.*tower_phi);
	s2 = cos(2.*tower_phi);    
	s13 = sin(3.*tower_phi);
	s23 = cos(3.*tower_phi);
	s14 = sin(4.*tower_phi);
	s24 = cos(4.*tower_phi);
	s15 = sin(5.*tower_phi);
	s25 = cos(5.*tower_phi);
	s16 = sin(6.*tower_phi);
	s26 = cos(6.*tower_phi);
	
	if(tower_energyet<minet_) continue;
	if(tower_energyet>maxet_) continue;;

	rp[etEcal    ]->addParticle (tower_energyet_e, s1,    s2,    tower_eta);
	rp[etEcalP   ]->addParticle (tower_energyet_e, s1,    s2,    tower_eta);
	rp[etEcalM   ]->addParticle (tower_energyet_e, s1,    s2,    tower_eta);
	rp[etHcal    ]->addParticle (tower_energyet_h, s1,    s2,    tower_eta);
	rp[etHcalP   ]->addParticle (tower_energyet_h, s1,    s2,    tower_eta);
	rp[etHcalM   ]->addParticle (tower_energyet_h, s1,    s2,    tower_eta);
	rp[etHF      ]->addParticle (tower_energyet,   s1,    s2,    tower_eta);
	rp[etHFp     ]->addParticle (tower_energyet,   s1,    s2,    tower_eta);
	rp[etHFm     ]->addParticle (tower_energyet,   s1,    s2,    tower_eta);
	rp[etCaloHFP ]->addParticle (tower_energyet,   s1,    s2,    tower_eta);
	rp[etCaloHFM ]->addParticle (tower_energyet,   s1,    s2,    tower_eta);

	rp[etHF3     ]->addParticle (tower_energyet,   s13,   s23,   tower_eta);
	rp[etHFp3    ]->addParticle (tower_energyet,   s13,   s23,   tower_eta);    
	rp[etHFm3    ]->addParticle (tower_energyet,   s13,   s23,   tower_eta);

	rp[etHF4     ]->addParticle (tower_energyet,   s14,   s24,   tower_eta);
	rp[etHFp4    ]->addParticle (tower_energyet,   s14,   s24,   tower_eta);    
	rp[etHFm4    ]->addParticle (tower_energyet,   s14,   s24,   tower_eta);

	rp[etHF5     ]->addParticle (tower_energyet,   s15,   s25,   tower_eta);
	rp[etHFp5    ]->addParticle (tower_energyet,   s15,   s25,   tower_eta);    
	rp[etHFm5    ]->addParticle (tower_energyet,   s15,   s25,   tower_eta);

	rp[etHF6     ]->addParticle (tower_energyet,   s16,   s26,   tower_eta);
	rp[etHFp6    ]->addParticle (tower_energyet,   s16,   s26,   tower_eta);    
	rp[etHFm6    ]->addParticle (tower_energyet,   s16,   s26,   tower_eta);
      } 
    }
    //Tracking part
    
    double track_eta;
    double track_phi;
    double track_pt;
#ifdef TRACKCOLLECTION  
    Handle<reco::TrackCollection> tracks;
    iEvent.getByLabel(trackCollection_, tracks);
    
    if(tracks.isValid()){
      for(reco::TrackCollection::const_iterator j = tracks->begin(); j != tracks->end(); j++){
	
	//Find possible collections with command line: edmDumpEventContent *.root
#endif
#ifdef RECOCHARGEDCANDIDATECOLLECTION
	edm::Handle<reco::RecoChargedCandidateCollection> trackCollection;
	iEvent.getByLabel("allMergedPtSplit12Tracks",trackCollection);
	//	iEvent.getByLabel("hiGoodMergedTracks",trackCollection);
	if(trackCollection.isValid()){
	  
	  const reco::RecoChargedCandidateCollection * tracks = trackCollection.product();
	  for(reco::RecoChargedCandidateCollection::const_iterator j = tracks->begin(); j != tracks->end(); j++){
#endif  
	edm::Handle<reco::VertexCollection> vertex;
	iEvent.getByLabel(vtxCollection_, vertex);
	
// find the vertex point and error

	math::XYZPoint vtxPoint(0.0,0.0,0.0);
	double vzErr =0.0, vxErr=0.0, vyErr=0.0;
	if(vertex->size()>0) {
	  vtxPoint=vertex->begin()->position();
	  vzErr=vertex->begin()->zError();
	  vxErr=vertex->begin()->xError();
	  vyErr=vertex->begin()->yError();
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
	if( isPixel )
	  {
	    // dz significance cut 
	    if ( fabs(dz/dzsigma) > dzerr_ ) accepted = false;
	    
	    // chi2/ndof cut 
	    if ( j->normalizedChi2() > chi2_ ) accepted = false;
	  }
	
	// cuts for full tracks
	if ( ! isPixel)
	  {
	    // dz and d0 significance cuts 
	    if ( fabs(dz/dzsigma) > 3 ) accepted = false;
	    if ( fabs(d0/d0sigma) > 3 ) accepted = false;
	    
	    // pt resolution cut
	    if ( j->ptError()/j->pt() > 0.05 ) accepted = false;
	    
	    // number of valid hits cut
	    if ( j->numberOfValidHits() < 12 ) accepted = false;
	  }
	if( accepted ) {
	  track_eta = j->eta();
	  track_phi = j->phi();
	  track_pt = j->pt();
	  double s =sin(2*track_phi);
	  double c =cos(2*track_phi);
	  double s3 =sin(3*track_phi);
	  double c3 =cos(3*track_phi);
	  double s4 =sin(4*track_phi);
	  double c4 =cos(4*track_phi);
	  double s5 =sin(5*track_phi);
	  double c5 =cos(5*track_phi);
	  double s6 =sin(6*track_phi);
	  double c6 =cos(6*track_phi);
	  double w = 1;
	  if(useTrackPtWeight_) {
	    w = track_pt;
	    if(w>2.5) w=2.0;   //v2 starts decreasing above ~2.5 GeV/c
	  }
	  if(track_pt<minpt_) continue;
	  if(track_pt>maxpt_) continue;
	  rp[EvtPlaneFromTracksMidEta]->addParticle(w,s,c,track_eta);
	  rp[EvtPTracksPosEtaGap]->addParticle(w,s,c,track_eta);
	  rp[EvtPTracksNegEtaGap]->addParticle(w,s,c,track_eta);
	  rp[EPTracksMid3]->addParticle(w,s3,c3,track_eta);
	  rp[EPTracksPos3]->addParticle(w,s3,c3,track_eta);
	  rp[EPTracksNeg3]->addParticle(w,s3,c3,track_eta);
	  rp[EPTracksMid4]->addParticle(w,s4,c4,track_eta);
	  rp[EPTracksPos4]->addParticle(w,s4,c4,track_eta);
	  rp[EPTracksNeg4]->addParticle(w,s4,c4,track_eta);
	  rp[EPTracksMid5]->addParticle(w,s5,c5,track_eta);
	  rp[EPTracksPos5]->addParticle(w,s5,c5,track_eta);
	  rp[EPTracksNeg5]->addParticle(w,s5,c5,track_eta);
	  rp[EPTracksMid6]->addParticle(w,s6,c6,track_eta);
	  rp[EPTracksPos6]->addParticle(w,s6,c6,track_eta);
	  rp[EPTracksNeg6]->addParticle(w,s6,c6,track_eta);
	  
	}
	
      }
    }
      }
      std::auto_ptr<EvtPlaneCollection> evtplaneOutput(new EvtPlaneCollection);
      
  EvtPlane *ep[NumEPNames];
  
  double ang=-10;
  double sv = 0;
  double cv = 0;

  for(int i = 0; i<NumEPNames; i++) {
    rp[i]->getAngle(ang,sv,cv);
    if(storeNames_) ep[i] = new EvtPlane(ang,sv,cv,EPNames[i]);
    else ep[i] = new EvtPlane(ang,sv,cv,"");
  }
  if(useTrack_) {
    for(int i = 0; i<9; i++) {
      evtplaneOutput->push_back(*ep[i]);
    }  
  }
  for(int i = 9; i<NumEPNames; i++) {
    if(useECAL_ && !useHCAL_) {
      if(EPNames[i].rfind("Ecal")!=string::npos) {
	evtplaneOutput->push_back(*ep[i]);
      }
    } else if (useHCAL_ && !useECAL_) {
      if(EPNames[i].rfind("Hcal")!=string::npos) {
	evtplaneOutput->push_back(*ep[i]);
      }
    }else if (useECAL_ && useHCAL_) {
      evtplaneOutput->push_back(*ep[i]);
    }
  }
  
  iEvent.put(evtplaneOutput, "recoLevel");
  //  storeNames_ = 0;
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
