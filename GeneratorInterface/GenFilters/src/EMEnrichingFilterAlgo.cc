#include "GeneratorInterface/GenFilters/interface/EMEnrichingFilterAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CLHEP/Vector/LorentzVector.h"


using namespace edm;
using namespace std;


EMEnrichingFilterAlgo::EMEnrichingFilterAlgo(const edm::ParameterSet& iConfig) { 

  //set constants
  FILTER_TKISOCUT_=4;
  FILTER_CALOISOCUT_=7;
  FILTER_ETA_MIN_=0;
  FILTER_ETA_MAX_=2.4;
  ECALBARRELMAXETA_=1.479;
  ECALBARRELRADIUS_=129.0;
  ECALENDCAPZ_=304.5;

  seedThresholdBarrel_=(float) iConfig.getParameter<double>("seedThresholdBarrel");
  clusterThresholdBarrel_=(float) iConfig.getParameter<double>("clusterThresholdBarrel");
  coneSizeBarrel_=(float) iConfig.getParameter<double>("coneSizeBarrel");
  seedThresholdEndcap_=(float) iConfig.getParameter<double>("seedThresholdEndcap");
  clusterThresholdEndcap_=(float) iConfig.getParameter<double>("clusterThresholdEndcap");
  coneSizeEndcap_=(float) iConfig.getParameter<double>("coneSizeEndcap");
  isoGenParETMin_=(float) iConfig.getParameter<double>("isoGenParETMin");
  isoGenParConeSize_=(float) iConfig.getParameter<double>("isoGenParConeSize");
  

}

EMEnrichingFilterAlgo::~EMEnrichingFilterAlgo() {
}


bool EMEnrichingFilterAlgo::filter(const edm::Event& iEvent, const edm::EventSetup& iSetup)  {

//   cout <<  seedThresholdBarrel_<<endl;
//   cout <<  clusterThresholdBarrel_<<endl;
//   cout <<  coneSizeBarrel_<<endl;
//   cout <<  seedThresholdEndcap_<<endl;
//   cout <<  clusterThresholdEndcap_<<endl;
//   cout <<  coneSizeEndcap_<<endl;
//   cout <<  isoGenParETMin_<<endl;
//   cout <<  isoGenParConeSize_<<endl;


  Handle<reco::GenParticleCollection> genParsHandle;
  iEvent.getByLabel("genParticles",genParsHandle);
  reco::GenParticleCollection genPars=*genParsHandle;

  //bending of traj. of charged particles under influence of B-field
  std::vector<reco::GenParticle> genParsCurved=applyBFieldCurv(genPars,iSetup);

    bool result1=filterPhotonElectronSeed(seedThresholdBarrel_,
				       clusterThresholdBarrel_,
				       coneSizeBarrel_,
				       seedThresholdEndcap_,
				       clusterThresholdEndcap_,
				       coneSizeEndcap_,
				       genParsCurved);

  bool result2=filterIsoGenPar(isoGenParETMin_,isoGenParConeSize_,genPars,genParsCurved);


  bool result=result1 || result2;
  
  return result;

}



//filter that uses clustering around photon and electron seeds
//only electrons, photons, charged pions, and charged kaons are clustered
//if a cluster is found with eT above the given threshold, the function returns true
//seed threshold, total threshold, and cone size/shape are specified separately for the barrel and the endcap
//if "conesize" argument is given as -1, a strip elongated in phi is used
//in the endcap a strip in phi is not possible, only a cone in xy
//the strip is 0.4 in phi by 0.04 in eta
bool EMEnrichingFilterAlgo::filterPhotonElectronSeed(float seedthreshold, 
						     float clusterthreshold, 
						     float conesize,
						     float seedthresholdendcap, 
						     float clusterthresholdendcap, 
						     float conesizeendcap,
						     const std::vector<reco::GenParticle> &genPars) {
  
  bool retval=false;
  
  vector<reco::GenParticle> seeds;
  //find electron and photon seeds - must have E>seedthreshold GeV
  for (uint32_t is=0;is<genPars.size();is++) {
    reco::GenParticle gp=genPars.at(is);
    if (gp.status()!=1 || fabs(gp.eta()) > FILTER_ETA_MAX_ || fabs(gp.eta()) < FILTER_ETA_MIN_) continue;
    int absid=abs(gp.pdgId());
    if (absid!=11 && absid!=22) continue;
    if (gp.et()>seedthreshold) seeds.push_back(gp);
  }
  
  //for every seed, try to cluster stable particles about it in cone of specified size  
  for (uint32_t is=0;is<seeds.size();is++) {
    float eInCone=0;
    bool isBarrel=fabs(seeds.at(is).eta())<ECALBARRELMAXETA_;
    for (uint32_t ig=0;ig<genPars.size();ig++) {
      reco::GenParticle gp=genPars.at(ig);
      if (gp.status()!=1) continue;
      int gpabsid=abs(gp.pdgId());
      if (gpabsid!=22 && gpabsid!=11 && gpabsid != 211 && gpabsid!= 321) continue;
      if (gp.energy()<5) continue;
      //treat barrel and endcap differently      
      if (isBarrel) {
	float dr=deltaR(seeds.at(is),gp);
	float dphi=deltaPhi(seeds.at(is).phi(),gp.phi());
	float deta=fabs(seeds.at(is).eta()-gp.eta());
	if (dr<conesize || (conesize<0 && deta<0.02 && dphi<0.2)) eInCone+=gp.et();
      } else {
	float drxy=deltaRxyAtEE(seeds.at(is),gp);
	if (drxy<conesizeendcap) {
	  eInCone+=gp.et();
	}
      }
    }
    if (isBarrel && eInCone>clusterthreshold) {
      retval=true;
      break;
    }
    if (!isBarrel && eInCone>clusterthresholdendcap){
      retval=true;
      break;
    }
  }
  
  return retval;
}


//make new genparticles vector taking into account the bending of charged particles in the b field
//only stable-final-state (status==1) particles, with ET>=1 GeV, have their trajectories bent
std::vector<reco::GenParticle> EMEnrichingFilterAlgo::applyBFieldCurv(const std::vector<reco::GenParticle> &genPars, const edm::EventSetup& iSetup) {
  
  
  vector<reco::GenParticle> curvedPars;

  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);

  Cylinder::CylinderPointer theBarrel=Cylinder::build(Surface::PositionType(0,0,0),Surface::RotationType(),ECALBARRELRADIUS_);
  Plane::PlanePointer endCapPlus=Plane::build(Surface::PositionType(0,0,ECALENDCAPZ_),Surface::RotationType());
  Plane::PlanePointer endCapMinus=Plane::build(Surface::PositionType(0,0,-1*ECALENDCAPZ_),Surface::RotationType());

  AnalyticalPropagator propagator(&(*magField),alongMomentum);  

  for (uint32_t ig=0;ig<genPars.size();ig++) {
    reco::GenParticle gp=genPars.at(ig);
    //don't bend trajectories of neutral particles, unstable particles, particles with < 1 GeV
    //particles with < ~0.9 GeV don't reach the barrel
    //so just put them as-is into the new vector
    if (gp.charge()==0 || gp.status()!=1 || gp.et()<1) {
      curvedPars.push_back(gp);
      continue;
    }
    GlobalPoint vertex(gp.vx(),gp.vy(),gp.vz());
    GlobalVector gvect(gp.px(),gp.py(),gp.pz());
    FreeTrajectoryState fts(vertex,gvect,gp.charge(),&(*magField));
    TrajectoryStateOnSurface impact;
    //choose to propagate to barrel, +Z endcap, or -Z endcap, according to eta
    if (fabs(gp.eta())<ECALBARRELMAXETA_) {
      impact=propagator.propagate(fts,*theBarrel);
    } else if (gp.eta()>0) {
      impact=propagator.propagate(fts,*endCapPlus);
    } else {
      impact=propagator.propagate(fts,*endCapMinus);
    }
    //in case the particle doesn't reach the barrel/endcap, just put it as-is into the new vector
    //it should reach though.
    if (!impact.isValid()) {
      curvedPars.push_back(gp);
      continue;
    }
    math::XYZTLorentzVector newp4;

    //the magnitude of p doesn't change, only the direction
    //NB I do get some small change in magnitude by the following,
    //I think it is a numerical precision issue
    float et=gp.et();
    float phinew=impact.globalPosition().phi();
    float pxnew=et*cos(phinew);
    float pynew=et*sin(phinew);
    newp4.SetPx(pxnew);
    newp4.SetPy(pynew);
    newp4.SetPz(gp.pz());
    newp4.SetE(gp.energy());
    reco::GenParticle gpnew=gp;
    gpnew.setP4(newp4);
    curvedPars.push_back(gpnew);
  }
  return curvedPars;


}

//calculate the difference in the xy-plane positions of gp1 and gp1 at the ECAL endcap
//if they go in different z directions returns 9999.
float EMEnrichingFilterAlgo::deltaRxyAtEE(const reco::GenParticle &gp1, const reco::GenParticle &gp2) {
  
  if (gp1.pz()*gp2.pz() < 0) return 9999.;
  
  float rxy1=ECALENDCAPZ_*tan(gp1.theta());
  float x1=cos(gp1.phi())*rxy1;
  float y1=sin(gp1.phi())*rxy1;

  float rxy2=ECALENDCAPZ_*tan(gp2.theta());
  float x2=cos(gp2.phi())*rxy2;
  float y2=sin(gp2.phi())*rxy2;
  
  float dxy=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
  return dxy;


}


//filter looking for isolated charged pions, charged kaons, and electrons.
//isolation done in cone of given size, looking at charged particles and neutral hadrons
//photons aren't counted in the isolation requirements

//need to have both the the curved and uncurved genpar collections
//because tracker iso has to be treated differently than calo iso
bool EMEnrichingFilterAlgo::filterIsoGenPar(float etmin, float conesize,const reco::GenParticleCollection &gph,
					const reco::GenParticleCollection &gphCurved
					) {
  
  for (uint32_t ip=0;ip<gph.size();ip++) {

    reco::GenParticle gp=gph.at(ip);
    reco::GenParticle gpCurved=gphCurved.at(ip);    
    int gpabsid=abs(gp.pdgId());
    //find potential candidates
    if (gp.et()<=etmin || gp.status()!=1) continue;
    if (gpabsid!=11 && gpabsid != 211 && gpabsid!= 321) continue;
    if (fabs(gp.eta()) < FILTER_ETA_MIN_) continue;
    if (fabs(gp.eta()) > FILTER_ETA_MAX_) continue;
    
    //check isolation
    float tkiso=0;
    float caloiso=0;
    for (uint32_t jp=0;jp<gph.size();jp++) {
      if (jp==ip) continue;
      reco::GenParticle pp=gph.at(jp);
      reco::GenParticle ppCurved=gphCurved.at(jp);
      if (pp.status() != 1) continue;
      float dr=deltaR(gp,pp);
      float drCurved=deltaR(gpCurved,ppCurved);
      if (abs(pp.charge())==1 && pp.et()>2 && dr<conesize) {
	tkiso+=pp.et();
      }
      if (pp.et()>2 && abs(pp.pdgId())!=22 && drCurved<conesize) {
	caloiso+=pp.energy();
      }
    }
    if (tkiso<FILTER_TKISOCUT_ && caloiso<FILTER_CALOISOCUT_) return true;
  }
  return false;
}

