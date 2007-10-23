#ifndef ValidationMisalignedTracker_h
#define ValidationMisalignedTracker_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// 
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "Math/GenVector/BitReproducible.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "DataFormats/Common/interface/RefVector.h"
#include "TrackingTools/GeomPropagators/interface/HelixExtrapolatorToLine2Order.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 


#include "TTree.h"
#include "TFile.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <cmath>
#include "TStyle.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// class decleration
//

class ValidationMisalignedTracker : public edm::EDAnalyzer {
public:
  explicit ValidationMisalignedTracker(const edm::ParameterSet&);
  ~ValidationMisalignedTracker();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  string simobject,trackassociator;
  bool selection_eff,selection_fake,ZmassSelection_;
  string rootfile_;
  
  bool skip;
  int eventCount_;
  TTree* tree_eff;
  TTree* tree_fake;
  TFile* file_;
  
  
  int irun, ievt;
 
  float mzmu,recmzmu,ptzmu,recptzmu,etazmu,thetazmu,phizmu,recetazmu,recthetazmu,recphizmu;
  float recenezmu, enezmu, pLzmu, recpLzmu,yzmu,recyzmu,mxptmu,recmxptmu,minptmu,recminptmu;
  int countpart[2],countpartrec[2];
  int flag,flagrec,count,countrec;
  // int countsim;
  float ene[2][2],p[2][2],px[2][2],py[2][2],pz[2][2],ptmu[2][2];
  float recene[2][2],recp[2][2],recpx[2][2],recpy[2][2],recpz[2][2],recptmu[2][2];
  
  int  trackType;
  float pt, eta, cottheta, theta, costheta, phi, d0, z0; 
  int nhit;
  float recpt, receta, rectheta, reccottheta, recphi, recd0, recz0;
  int nAssoc, recnhit;
  float recchiq;
  float reseta, respt, resd0, resz0, resphi,rescottheta,eff;
  
  float fakemzmu,fakerecmzmu,fakeptzmu,fakerecptzmu,fakeetazmu,fakethetazmu,fakephizmu,fakerecetazmu,fakerecthetazmu,fakerecphizmu;
  float fakerecenezmu, fakeenezmu, fakepLzmu, fakerecpLzmu,fakeyzmu,fakerecyzmu,fakemxptmu,fakerecmxptmu,fakeminptmu,fakerecminptmu;
  int fakecountpart[2],fakecountpartrec[2],fakeflag,fakeflagrec,fakecount,fakecountsim,fakecountrec;
  float fakeene[2][2],fakep[2][2],fakepx[2][2],fakepy[2][2],fakepz[2][2],fakeptmu[2][2];
  float fakerecene[2][2],fakerecp[2][2],fakerecpx[2][2],fakerecpy[2][2],fakerecpz[2][2],fakerecptmu[2][2];
  
  int  faketrackType;
  float fakept, fakeeta, fakecottheta, faketheta, fakecostheta, fakephi, faked0, fakez0; 
  int fakenhit;
  float fakerecpt, fakereceta, fakerectheta, fakereccottheta, fakerecphi, fakerecd0, fakerecz0;
  int fakenAssoc, fakerecnhit;
  float fakerecchiq;
  float fakereseta, fakerespt, fakeresd0, fakeresz0, fakeresphi,fakerescottheta,fake;
  
  HepLorentzVector vertexPosition;

  double chi2tmp;
  float fractiontmp;
  bool onlyDiag;
  edm::ESHandle<MagneticField> theMF;
  std::vector<std::string> associators;
  std::vector<const TrackAssociatorBase*> associatore;
 
  std::vector<edm::InputTag> label;
  edm::InputTag label_tp_effic;
  edm::InputTag label_tp_fake;

  GlobalVector magField;
  vector<float> ptused;



};

#endif

