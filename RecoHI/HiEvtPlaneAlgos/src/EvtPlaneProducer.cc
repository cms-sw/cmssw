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
// $Id: EvtPlaneProducer.cc,v 1.9 2011/09/20 18:07:31 ssanders Exp $
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

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <cstdlib>
#include "TROOT.h"
//#include "TFile.h"
#include "TH1.h"
#include "TH2D.h"
#include "TH2F.h"
#include "TTree.h"
#include "TH1I.h"
#include "TF1.h"
#include "TRandom3.h"
#include "TRandom3.h"

using std::rand;

using namespace std;

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"

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
      sumsin_sub1 = 0;
      sumsin_sub2 = 0;
      sumcos_sub1 = 0;
      sumcos_sub2 = 0;
      order = (double) orderval;
      ran = new TRandom3();
      ranSubEvnt = false;
      etaSubEvnt = true;
      etaGap = 0.4;
      nB = 4;
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
	if(ranSubEvnt) {
	  if(ran->Uniform(0.,1.) < 0.5) {
	    sumsin_sub1+=w*s;
	    sumcos_sub1+=w*c;
	    mult_sub1+=w;
	  } else {
	    sumsin_sub2+=w*s;
	    sumcos_sub2+=w*c;
	    mult_sub2+=w;
	  }
	}
	if(etaSubEvnt) {
	  int isub = FindSubEvent(eta);
	  if(isub == 0) {
	    sumsin_sub1+=w*s;
	    sumcos_sub1+=w*c;
	    mult_sub1+=w;
	  } else if (isub == 1) {
	    sumsin_sub2+=w*s;
	    sumcos_sub2+=w*c;
	    mult_sub2+=w;
	  }
	}
      }
    }
    Int_t FindSubEvent(Double_t eta) {
      Double_t shift = 0;
      Double_t DellRange = etamax2-etamin2+etamax1-etamin1;
      Double_t DellBin = DellRange/nB; 
      Double_t reta = eta-etamin1;
      if (etamax2>etamin2 && eta> etamax1) {
	if(eta>etamax1 && eta<etamin2) return -1;
	shift = etamin2-etamax1;
	reta-=shift;
      }
      if(reta < 0 || reta>DellRange) return -1;
      Int_t b = (Int_t ) (reta/DellBin);
      if(reta-b*DellBin>DellBin-etaGap/2. && 
	 b!=nB-1 && etamax1-eta>etaGap/2.) {
	return -1;
      }
      if(reta-b*DellBin<etaGap/2. && b!=0) {
	if(etamax2!=etamin2 && eta-etamin2>etaGap/2.) {
	  return -1;
	}
      }
      double db = b;
      int iretval = -1;
      if(etamin2 == etamax2 || eta<etamax1 || 
	 nB==2 ||(Int_t) (fmod(db/2.,2.e0)+0.1) ==1) {
	iretval =  (Int_t) (fmod(db,2.e0)+0.1);
      } else {
	iretval = (Int_t) (fmod(db+1.,2.e0)+0.1);
      }
      return iretval;
    }
    
    double getAngle(int sub, double &ang, double &sv, double &cv, double &Q, int &retmult){
      ang = -10;
      sv = 0;
      cv = 0;
      Q = 0;
      retmult = 0;
      if(sub == 0) {
	sv = sumsin;
	cv = sumcos;
	retmult = mult;
      } else if (sub == 1) {
	sv = sumsin_sub1;
	cv = sumcos_sub1;
	retmult = mult_sub1;
      } else if (sub == 2) {
	sv = sumsin_sub2;
	cv = sumcos_sub2;
	retmult = mult_sub2;
      }

      Q = sv*sv+cv*cv;
      if(Q>0) {
	ang = atan2(sv,cv)/order;
	Q = sqrt(Q);
      }
 
      return ang;
    }
    void reset() {
      sumsin=0;
      sumcos=0;
      sumsin_sub1 = 0;
      sumsin_sub2 = 0;
      sumcos_sub1 = 0;
      sumcos_sub2 = 0;
      mult = 0;
      mult_sub1 = 0;
      mult_sub2 = 0;
    }
  private:
    string epname;
    double etamin1;
    double etamax1;

    double etamin2;
    double etamax2;
    double sumsin;
    double sumcos;
    double sumsin_sub1;
    double sumsin_sub2;
    double sumcos_sub1;
    double sumcos_sub2;
    int mult;
    int mult_sub1;
    int mult_sub2;
    TRandom3 * ran;
    double order;
    bool etaSubEvnt;
    bool ranSubEvnt;
    double etaGap;
    int nB;
  };
  

  GenPlane *rp[NumEPNames];

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  bool useECAL_;
  bool useHCAL_;
  bool genSubEvt_;
  bool useTrack_;
  bool useTrackPtWeight_;
  double minet_;
  double maxet_;
  double minpt_;
  double maxpt_;
  double minvtx_;
  double maxvtx_;
  double biggap_;
  double dzerr_;
  double chi2_;
  TRandom3 * ran;
  TH1D * etSpec;
  TH1D * ptSpec;
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
  
  int k=100;
  int iseed=k;
  
  srand(iseed);
  ran = new TRandom3();
  //register your products
  
  genSubEvt_ = iConfig.getUntrackedParameter<bool>("genSubEvt",false);
  useECAL_ = iConfig.getUntrackedParameter<bool>("useECAL",true);
  useHCAL_ = iConfig.getUntrackedParameter<bool>("useHCAL",true);
  useTrack_ = iConfig.getUntrackedParameter<bool>("useTrack",true);
  useTrackPtWeight_ = iConfig.getUntrackedParameter<bool>("useTrackPtWeight",true);
  minet_ = iConfig.getUntrackedParameter<double>("minet_",0.2);
  maxet_ = iConfig.getUntrackedParameter<double>("maxet_",500.);
  minpt_ = iConfig.getUntrackedParameter<double>("minpt_",0.3);
  maxpt_ = iConfig.getUntrackedParameter<double>("maxpt_",2.5);
  minvtx_ = iConfig.getUntrackedParameter<double>("minvtx_",-50.);
  maxvtx_ = iConfig.getUntrackedParameter<double>("maxvtx_",50.);
  biggap_ = iConfig.getUntrackedParameter<double>("biggap_",0.3);
  dzerr_ = iConfig.getUntrackedParameter<double>("dzerr_",14.);
  chi2_  = iConfig.getUntrackedParameter<double>("chi2_",80.);

  //TFileDirectory subdir = fs->mkdir("EvtPlaneProducer");
  //etSpec = subdir.make<TH1D>("etSpec","etSpec",840,-20,400);
  //ptSpec = subdir.make<TH1D>("ptSpec","ptSpec",200,0,50);
  //cout<<"////////////////////////  EvtPlaneProducer /////////////////"<<endl;
  //if(genSubEvt_) 
//     cout<<"genSubEvt_ true"<<endl;
//   else
//     cout<<"genSubEvt_ false"<<endl;
//   if(useECAL_) 
//     cout<<"useECAL_ true"<<endl;
//   else
//     cout<<"useECAL_ false"<<endl;
//   if(useHCAL_) 
//     cout<<"useHCAL_ true"<<endl;
//   else
//     cout<<"useHCAL_ false"<<endl;
//   if(useTrack_) 
//     cout<<"useTrack_ true"<<endl;
//   else
//     cout<<"useTrack_ false"<<endl;
//   if(useTrackPtWeight_) 
//     cout<<"useTrackPtWeight_ true"<<endl;
//   else
//     cout<<"useTrackPtWeight_ false"<<endl;


  produces<reco::EvtPlaneCollection>("recoLevel");
  Double_t g1low = 0;
  Double_t g1high = 0;
  Double_t g2low = 0;
  Double_t g2high = 0;
  if(biggap_>0) {
    Double_t ernge = (4.-2.*biggap_)/3.;
    g1low = -2. + ernge;
    g1high = g1low+biggap_;
    g2low = g1high+ernge;
    g2high = 2.-ernge;
  }
  for(int i = 0; i<NumEPNames; i++ ) {
    //if(i==EPTracksPosEtaBigGap&&biggap_>0){
      //rp[i] = new GenPlane(EPNames[i],g2high,2.,0,0,EPOrder[i]);
    //} else if (i==EPTracksNegEtaBigGap&&biggap_>0) {
    //  rp[i] = new GenPlane(EPNames[i],-2.,g1low,0,0,EPOrder[i]);
    //} else if (i==EvtPlaneFromTracksMidEta&&biggap_>0) {
    //  rp[i] = new GenPlane(EPNames[i],g1high,g2low,0,0,EPOrder[i]);
    //} else {
      rp[i] = new GenPlane(EPNames[i],EPEtaMin1[i],EPEtaMax1[i],EPEtaMin2[i],EPEtaMax2[i],EPOrder[i]);
      //}
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
  using namespace HepMC;
  int vs_sell;
  float vzr_sell;
  float vzErr_sell;
  //
  //Get Vertex
  //
  edm::Handle<reco::VertexCollection> vertexCollection3;
  iEvent.getByLabel("hiSelectedVertex",vertexCollection3);
  const reco::VertexCollection * vertices3 = vertexCollection3.product();
  vs_sell = vertices3->size();
  if(vs_sell>0) {
    vzr_sell = vertices3->begin()->z();
    vzErr_sell = vertices3->begin()->zError();
  } else
    vzr_sell = -999.9;
  //
  for(int i = 0; i<NumEPNames; i++) rp[i]->reset();
  if(vzr_sell>minvtx_ && vzr_sell<maxvtx_) {
    //calorimetry part
    
    double tower_eta, tower_phi;
    double tower_energy, tower_energy_e, tower_energy_h;
    double tower_energyet, tower_energyet_e, tower_energyet_h;
    double s1, s2, s11, s21,s13,s23,s14,s24,s15,s25,s16,s26;
    Handle<CaloTowerCollection> calotower;
    iEvent.getByLabel("towerMaker",calotower);
    
    if(calotower.isValid()){
      
      for (CaloTowerCollection::const_iterator j = calotower->begin();j !=calotower->end(); j++) {   
	tower_eta        = j->eta();
	tower_phi        = j->phi();
	tower_energy_e   = j->emEnergy();
	tower_energy_h   = j->hadEnergy();
	tower_energy     = tower_energy_e + tower_energy_h;
	tower_energyet_e   = j->emEt();
	tower_energyet_h   = j->hadEt();
	tower_energyet     = tower_energyet_e + tower_energyet_h;
	
	s1 = sin(2.*tower_phi);
	s2 = cos(2.*tower_phi);    
	s11 = sin(tower_phi);
	s21 = cos(tower_phi);
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
	//etSpec->Fill(tower_energyet);

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
    double track_charge;
#ifdef TRACKCOLLECTION  
    Handle<reco::TrackCollection> tracks;
    //    iEvent.getByLabel("hiSelectedTracks", tracks);
    iEvent.getByLabel("hiGoodTightMergedTracks", tracks);
    
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
	iEvent.getByLabel("hiSelectedVertex", vertex);
	
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
	  track_charge = j->charge();
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
	  //ptSpec->Fill(track_pt);
	  //	  if(track_charge > 0 ){
	  //	    rp[EvtPlaneFromTracksPosCh]->addParticle(w,s,c,track_eta);
	  //	  } else if(track_charge < 0){
	  //   rp[EvtPlaneFromTracksNegCh]->addParticle(w,s,c,track_eta);
	  // }
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
  EvtPlane *ep1[NumEPNames];
  EvtPlane *ep2[NumEPNames];
  
  double ang=-10;
  double sv = 0;
  double cv = 0;
  double Q = 0;
  int mult = 0;

  for(int i = 0; i<NumEPNames; i++) {
    rp[i]->getAngle(0,ang,sv,cv,Q,mult);
    ep[i] = new EvtPlane(ang,sv,cv,EPNames[i].data());
    if(genSubEvt_) {
      rp[i]->getAngle(1,ang,sv,cv,Q,mult);
      ep1[i] = new EvtPlane(ang, sv, cv, Form("%s_sub1",EPNames[i].data()));
      rp[i]->getAngle(2,ang,sv,cv,Q,mult);
      ep2[i] = new EvtPlane(ang, sv, cv, Form("%s_sub2",EPNames[i].data()));
    }
  }
  if(useTrack_) {
    for(int i = 0; i<9; i++) {
      evtplaneOutput->push_back(*ep[i]);
      if(genSubEvt_) {
	evtplaneOutput->push_back(*ep1[i]);
	evtplaneOutput->push_back(*ep2[i]);
      }
    }  
  }
  for(int i = 9; i<NumEPNames; i++) {
    if(useECAL_ && !useHCAL_) {
      if(EPNames[i].rfind("Ecal")!=string::npos) {
	evtplaneOutput->push_back(*ep[i]);
	if(genSubEvt_) {
	  evtplaneOutput->push_back(*ep1[i]);
	  evtplaneOutput->push_back(*ep2[i]);
	}
      }
    } else if (useHCAL_ && !useECAL_) {
      if(EPNames[i].rfind("Hcal")!=string::npos) {
	evtplaneOutput->push_back(*ep[i]);
	if(genSubEvt_) {
	  evtplaneOutput->push_back(*ep1[i]);
	  evtplaneOutput->push_back(*ep2[i]);
	}
      }
    }else if (useECAL_ && useHCAL_) {
      evtplaneOutput->push_back(*ep[i]);
      if(genSubEvt_) {
	evtplaneOutput->push_back(*ep1[i]);
	evtplaneOutput->push_back(*ep2[i]);
      }
    }
  }
  
  iEvent.put(evtplaneOutput, "recoLevel");
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
