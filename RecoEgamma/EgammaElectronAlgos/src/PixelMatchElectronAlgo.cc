// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelMatchElectronAlgo.
// 
/**\class PixelMatchElectronAlgo EgammaElectronAlgos/PixelMatchElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking

*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Thu july 6 13:22:06 CEST 2006
// $Id: PixelMatchElectronAlgo.cc,v 1.26 2006/12/20 12:22:03 uberthon Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/GsfTools/interface/GSUtilities.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <sstream>

#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

PixelMatchElectronAlgo::PixelMatchElectronAlgo(double maxEOverPBarrel, double maxEOverPEndcaps, 
                                               double hOverEConeSize, double maxHOverE, 
                                               double maxDeltaEta, double maxDeltaPhi, double ptcut):  
  maxEOverPBarrel_(maxEOverPBarrel), maxEOverPEndcaps_(maxEOverPEndcaps), 
  hOverEConeSize_(hOverEConeSize), maxHOverE_(maxHOverE), 
  maxDeltaEta_(maxDeltaEta), maxDeltaPhi_(maxDeltaPhi), ptCut_(ptcut)
{   
  printf("Algo Constructor start===================\n");fflush(stdout);
  geomPropBw_=0;	
  geomPropFw_=0;	
  mtsTransform_=0;
  printf("Algo Constructor end===================\n");fflush(stdout);
}

PixelMatchElectronAlgo::~PixelMatchElectronAlgo() {
  printf("Algo Destructor start ===================\n");fflush(stdout);
  delete geomPropBw_;
  printf("Algo Destructor 1 ===================\n");fflush(stdout);
  delete geomPropFw_;
  printf("Algo Destructor 2 ===================\n");fflush(stdout);
  delete mtsTransform_;
  printf("Algo Destructor end ===================\n");fflush(stdout);
}

void PixelMatchElectronAlgo::setupES(const edm::EventSetup& es, const edm::ParameterSet &conf) {
  printf("Algo setupES start ===================\n");fflush(stdout);

  //services
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  es.get<IdealMagneticFieldRecord>().get(theMagField);
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle_);

  // get calo geometry
  es.get<IdealGeometryRecord>().get(theCaloGeom);
  
  mtsTransform_ = new MultiTrajectoryStateTransform;
  geomPropBw_ = new GsfPropagatorAdapter(AnalyticalPropagator(theMagField.product(), oppositeToMomentum));
  geomPropFw_ = new GsfPropagatorAdapter(AnalyticalPropagator(theMagField.product(), alongMomentum));

  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  trackBarrelLabel_ = conf.getParameter<string>("TrackBarrelLabel");
  trackBarrelInstanceName_ = conf.getParameter<string>("TrackBarrelProducer");
  trackEndcapLabel_ = conf.getParameter<string>("TrackEndcapLabel");
  trackEndcapInstanceName_ = conf.getParameter<string>("TrackEndcapProducer");
  assBarrelLabel_ = conf.getParameter<string>("SCLBarrelLabel");
  assBarrelInstanceName_ = conf.getParameter<string>("SCLBarrelProducer");
  assEndcapLabel_ = conf.getParameter<string>("SCLEndcapLabel");
  assEndcapInstanceName_ = conf.getParameter<string>("SCLEndcapProducer");
  assBarrelTrTSLabel_ = conf.getParameter<string>("AssocTrTSBarrelLabel");
  assBarrelTrTSInstanceName_ = conf.getParameter<string>("AssocTrTBarrelProducer");
  assEndcapTrTSLabel_ = conf.getParameter<string>("AssocTrTEndcapLabel");
  assEndcapTrTSInstanceName_ = conf.getParameter<string>("AssocTrTEndcapProducer");
  printf("Algo setupES end ===================\n");fflush(stdout);
}

void  PixelMatchElectronAlgo::run(Event& e, PixelMatchGsfElectronCollection & outEle) {

  // get the input 
  edm::Handle<GsfTrackCollection> tracksBarrelH;
  edm::Handle<GsfTrackCollection> tracksEndcapH;
  // to check existance
  edm::Handle<HBHERecHitCollection> hbhe;
  HBHERecHitMetaCollection *mhbhe=0;
  if (hOverEConeSize_ > 0.) {
    e.getByType(hbhe);  
    mhbhe=  &HBHERecHitMetaCollection(*hbhe);  //FIXME, generates warning
  }
  e.getByLabel(trackBarrelLabel_,trackBarrelInstanceName_,tracksBarrelH);
  e.getByLabel(trackEndcapLabel_,trackEndcapInstanceName_,tracksEndcapH);

  edm::Handle<SeedSuperClusterAssociationCollection> barrelH;
  edm::Handle<SeedSuperClusterAssociationCollection> endcapH;
  e.getByLabel(assBarrelLabel_,assBarrelInstanceName_,barrelH);
  e.getByLabel(assEndcapLabel_,assEndcapInstanceName_,endcapH);

  edm::Handle<GsfTrackSeedAssociationCollection> barrelTSAssocH;
  edm::Handle<GsfTrackSeedAssociationCollection> endcapTSAssocH;
  e.getByLabel(assBarrelTrTSLabel_,assBarrelTrTSInstanceName_,barrelTSAssocH);
  e.getByLabel(assEndcapTrTSLabel_,assEndcapTrTSInstanceName_,endcapTSAssocH);
  edm::LogInfo("") 
    <<"\n\n Treating "<<e.id()<<", Number of seeds Barrel:"
    <<barrelH.product()->size()<<" Number of seeds Endcap:"<<endcapH.product()->size();
  
  // create electrons from tracks in 2 steps: barrel + endcap
  const SeedSuperClusterAssociationCollection  *sclAss=&(*barrelH);
  process(tracksBarrelH,sclAss,barrelTSAssocH.product(),mhbhe,outEle);
  sclAss=&(*endcapH);
  process(tracksEndcapH,sclAss,endcapTSAssocH.product(),mhbhe,outEle);

  std::ostringstream str;

  str << "========== PixelMatchElectronAlgo Info ==========";
  str << "Event " << e.id();
  str << "Number of final electron tracks: " << tracksBarrelH.product()->size()+ tracksEndcapH.product()->size();
  str << "Number of final electrons: " << outEle.size();
  for (vector<PixelMatchGsfElectron>::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
    str << "New electron with charge, pt, eta, phi : "  << it->charge() << " , " 
        << it->pt() << " , " << it->eta() << " , " << it->phi();
  }
 
  str << "=================================================";
  LogDebug("PixelMatchElectronAlgo") << str.str();
  return;
}

void PixelMatchElectronAlgo::process(edm::Handle<GsfTrackCollection> tracksH, const SeedSuperClusterAssociationCollection *sclAss, const GsfTrackSeedAssociationCollection *tsAss,
                                     HBHERecHitMetaCollection *mhbhe, PixelMatchGsfElectronCollection & outEle) {
  const GsfTrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {
    const GsfTrack & t=(*tracks)[i];
    const GsfTrackRef trackRef = edm::Ref<GsfTrackCollection>(tracksH,i);
    edm::Ref<TrajectorySeedCollection> seed = (*tsAss)[trackRef];
    const SuperCluster theClus=*((*sclAss)[seed]);

    // calculate HoE
    double HoE;
    if (mhbhe) {
      CaloConeSelector sel(hOverEConeSize_, theCaloGeom.product(), DetId::Hcal);
      GlobalPoint pclu(theClus.x(),theClus.y(),theClus.z());
      double hcalEnergy = 0.;
      std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,*mhbhe);
      for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
	//std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
	hcalEnergy += i->energy();
      }
      HoE = hcalEnergy/theClus.energy();
      LogDebug("") << "H/E : " << HoE;
    } else HoE=0;


    // calculate Trajectory StatesOnSurface....
    //at innermost point
    TrajectoryStateOnSurface innTSOS = mtsTransform_->innerStateOnSurface(t, *(trackerHandle_.product()), theMagField.product());
 
    //at vertex
    // innermost state propagation to the nominal vertex
    TrajectoryStateOnSurface vtxTSOS =
      TransverseImpactPointExtrapolator(*geomPropBw_).extrapolate(innTSOS,GlobalPoint(0,0,0));
    if (!vtxTSOS.isValid()) vtxTSOS=innTSOS;

    //at seed
    TrajectoryStateOnSurface outTSOS = mtsTransform_->outerStateOnSurface(t, *(trackerHandle_.product()), theMagField.product());
    TrajectoryStateOnSurface seedTSOS = TransverseImpactPointExtrapolator(*geomPropFw_).extrapolate(outTSOS,GlobalPoint(theClus.seed()->position().x(),theClus.seed()->position().y(),theClus.seed()->position().z()));
 

    //at scl
    TrajectoryStateOnSurface sclTSOS = TransverseImpactPointExtrapolator(*geomPropFw_).extrapolate(innTSOS,GlobalPoint(theClus.x(),theClus.y(),theClus.z()));

    GlobalVector vtxMom=computeMode(vtxTSOS);
    GlobalPoint  sclPos=sclTSOS.globalPosition();
    if (preSelection(theClus,vtxMom, sclPos, HoE)) {
      GlobalVector innMom=computeMode(innTSOS);
      GlobalPoint innPos=innTSOS.globalPosition();
      GlobalVector seedMom=computeMode(seedTSOS);
      GlobalPoint  seedPos=seedTSOS.globalPosition();
      GlobalVector sclMom=computeMode(sclTSOS);    
      GlobalPoint  vtxPos=vtxTSOS.globalPosition();
      GlobalVector outMom=computeMode(outTSOS);
      GlobalPoint  outPos=outTSOS.globalPosition();

      PixelMatchGsfElectron ele((*sclAss)[seed],trackRef,sclPos,sclMom,seedPos,seedMom,innPos,innMom,vtxPos,vtxMom,outPos,outMom,HoE);
      // set corrections + classification
      ElectronClassification theClassifier;
      theClassifier.correct(ele);
      ElectronEnergyCorrector theEnCorrector;
      theEnCorrector.correct(ele);
      ElectronMomentumCorrector theMomCorrector;
      theMomCorrector.correct(ele,vtxTSOS);
	//mCorr.getBestMomentum(),mCorr.getSCEnergyError(),mCorr.getTrackMomentumError());
      outEle.push_back(ele);
      LogInfo("")<<"Constructed new electron with energy  "<< (*sclAss)[seed]->energy();
    }
  }  // loop over tracks
}

bool PixelMatchElectronAlgo::preSelection(const SuperCluster& clus, const GlobalVector& tsosVtxMom, const GlobalPoint& tsosSclPos, double HoE) 
{
  LogDebug("")<< "========== preSelection ==========";
 
  LogDebug("") << "E/p : " << clus.energy()/tsosVtxMom.mag();
  if (tsosVtxMom.perp()<ptCut_)   return false;
  //FIXME: how to get detId from a cluster??
  std::vector<DetId> vecId=clus.getHitsByDetId();
  int subdet =vecId[0].subdetId();  //FIXME: is the first one really the biggest??
  if ((subdet==EcalBarrel) && (clus.energy()/tsosVtxMom.mag() > maxEOverPBarrel_)) return false;
  if ((subdet==EcalEndcap) && (clus.energy()/tsosVtxMom.mag() > maxEOverPEndcaps_)) return false;
  LogDebug("") << "E/p criteria is satisfied ";
  // delta eta criteria
  double etaclu = clus.eta();
  double etatrk = tsosSclPos.eta();
  double deta = etaclu-etatrk;
  LogDebug("") << "delta eta : " << deta;
  if (fabs(deta) > maxDeltaEta_) return false;
  LogDebug("") << "Delta eta criteria is satisfied ";
  // delta phi criteria
  double phiclu = clus.phi();
  double phitrk = tsosSclPos.phi();
  double dphi = phiclu-phitrk;
  LogDebug("") << "delta phi : " << dphi;
  if (fabs(dphi) > maxDeltaPhi_) return false;
  LogDebug("") << "Delta phi criteria is satisfied ";

  if (HoE > maxHOverE_) return false; //FIXME: passe dans tous les cas?
  LogDebug("") << "H/E criteria is satisfied ";

  LogDebug("") << "electron has passed preselection criteria ";
  LogDebug("") << "=================================================";
  return true;  
}  

GlobalVector PixelMatchElectronAlgo::computeMode(const TrajectoryStateOnSurface &tsos) {
  // mode computation	
  float mode_Px = 0.;
  float mode_Py = 0.;
  float mode_Pz = 0.;
  if ( tsos.isValid() ){
	  
    int Count = 0;
    unsigned int numb = tsos.components().size();
    float *Wgt   = new float[numb];
    float *Px    = new float[numb];
    float *Py    = new float[numb];
    float *Pz    = new float[numb];
    float *PxErr = new float[numb];
    float *PyErr = new float[numb];
    float *PzErr = new float[numb];
	  
    for (unsigned int ii = 0; ii < numb; ii ++){
      Wgt[ii]   = 0.;
      Px[ii]    = 0.;
      Py[ii]    = 0.;
      Pz[ii]    = 0.;
      PxErr[ii] = 0.;
      PyErr[ii] = 0.;
      PzErr[ii] = 0.;
    }
	  
    std::vector<TrajectoryStateOnSurface> comp = tsos.components();
    for (std::vector<TrajectoryStateOnSurface>::const_iterator it_comp = comp.begin(); it_comp!= comp.end(); it_comp++){
      Wgt[Count]    = it_comp->weight();
      Px[Count]     = it_comp->globalMomentum().x();
      Py[Count]     = it_comp->globalMomentum().y();
      Pz[Count]     = it_comp->globalMomentum().z();
      PxErr[Count]  = sqrt(it_comp->cartesianError().matrix()[3][3]);
      PyErr[Count]  = sqrt(it_comp->cartesianError().matrix()[4][4]);
      PzErr[Count]  = sqrt(it_comp->cartesianError().matrix()[5][5]);
      Count++;
    }
	  
    GSUtilities *myGSUtil_Px = new GSUtilities(numb, Wgt, Px, PxErr);
    GSUtilities *myGSUtil_Py = new GSUtilities(numb, Wgt, Py, PyErr);
    GSUtilities *myGSUtil_Pz = new GSUtilities(numb, Wgt, Pz, PzErr);
	  
    mode_Px = myGSUtil_Px->mode();
    mode_Py = myGSUtil_Py->mode();
    mode_Pz = myGSUtil_Pz->mode();
	 
    if ( myGSUtil_Px ) { delete myGSUtil_Px; }
    if ( myGSUtil_Py ) { delete myGSUtil_Py; }
    if ( myGSUtil_Pz ) { delete myGSUtil_Pz; }
	  
    delete[] Wgt;
    delete[] Px;
    delete[] PxErr;
    delete[] Py;
    delete[] PyErr;
    delete[] Pz;
    delete[] PzErr;
  } else printf("tsos not valid!!\n");fflush(stdout);
  return GlobalVector(mode_Px,mode_Py,mode_Pz);	

}

