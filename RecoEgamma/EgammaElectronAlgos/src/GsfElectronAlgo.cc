// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      GsfElectronAlgo
// 
/**\class GsfElectronAlgo EgammaElectronAlgos/GsfElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking
 
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Thu july 6 13:22:06 CEST 2006
// $Id: GsfElectronAlgo.cc,v 1.7 2008/02/25 10:40:03 uberthon Exp $
//
//

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
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
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <sstream>
#include <Math/VectorUtil.h>
#include <Math/Point3D.h>


using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

GsfElectronAlgo::GsfElectronAlgo(const edm::ParameterSet& conf,
                                               double maxEOverPBarrel, double maxEOverPEndcaps, 
                                               double minEOverPBarrel, double minEOverPEndcaps,
                                               double hOverEConeSize, double maxHOverE, 
                                               double maxDeltaEta, double maxDeltaPhi, 
					       bool highPtPresel, double highPtMin,
   				               bool applyEtaCorrection):  
  maxEOverPBarrel_(maxEOverPBarrel), maxEOverPEndcaps_(maxEOverPEndcaps), 
  minEOverPBarrel_(minEOverPBarrel), minEOverPEndcaps_(minEOverPEndcaps), 
  hOverEConeSize_(hOverEConeSize), maxHOverE_(maxHOverE), 
  maxDeltaEta_(maxDeltaEta), maxDeltaPhi_(maxDeltaPhi),
  highPtPreselection_(highPtPresel), highPtMin_(highPtMin),
  applyEtaCorrection_(applyEtaCorrection)
{   
 // this is the new version allowing to configurate the algo
  // interfaces still need improvement!!
  geomPropBw_=0;	
  geomPropFw_=0;	
  mtsTransform_=0;

  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  hbheLabel_ = conf.getParameter<string>("hbheModule");
  hbheInstanceName_ = conf.getParameter<string>("hbheInstance");
  trackLabel_ = conf.getParameter<string>("TrackLabel");
  trackInstanceName_ = conf.getParameter<string>("TrackProducer");
  assBarrelShapeInstanceName_ = conf.getParameter<string>("AssocShapeBarrelProducer");
  assBarrelShapeLabel_ = conf.getParameter<string>("AssocShapeBarrelLabel");
  assEndcapShapeInstanceName_ = conf.getParameter<string>("AssocShapeEndcapProducer");
  assEndcapShapeLabel_ = conf.getParameter<string>("AssocShapeEndcapLabel");

  // get type of processing
  processType_=1;
  if (conf.getParameter<string>("ElectronType")=="GlobalGsfElectron") processType_=2;
}

GsfElectronAlgo::~GsfElectronAlgo() {
  delete geomPropBw_;
  delete geomPropFw_;
  delete mtsTransform_;
}

void GsfElectronAlgo::setupES(const edm::EventSetup& es) {

  //services
  es.get<IdealMagneticFieldRecord>().get(theMagField);
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle_);

  // get calo geometry
  es.get<IdealGeometryRecord>().get(theCaloGeom);
  
  if (mtsTransform_) delete mtsTransform_;
  mtsTransform_ = new MultiTrajectoryStateTransform;
  if (geomPropBw_) delete geomPropBw_;
  geomPropBw_ = new GsfPropagatorAdapter(AnalyticalPropagator(theMagField.product(), oppositeToMomentum));
  if (geomPropFw_) delete geomPropFw_;
  geomPropFw_ = new GsfPropagatorAdapter(AnalyticalPropagator(theMagField.product(), alongMomentum));

}

void  GsfElectronAlgo::run(Event& e, GsfElectronCollection & outEle) {

  // get the input 
  edm::Handle<GsfTrackCollection> tracksH;

  // to check existence
  edm::Handle<HBHERecHitCollection> hbhe;
  HBHERecHitMetaCollection *mhbhe=0;
  if (hOverEConeSize_ > 0.) {
    e.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);  
    mhbhe=  new HBHERecHitMetaCollection(*hbhe);
  }
  e.getByLabel(trackLabel_,trackInstanceName_,tracksH);
  
  edm::Handle<BasicClusterShapeAssociationCollection> barrelShapeAssocH;
  edm::Handle<BasicClusterShapeAssociationCollection> endcapShapeAssocH;
  e.getByLabel(assBarrelShapeLabel_,assBarrelShapeInstanceName_,barrelShapeAssocH);
  e.getByLabel(assEndcapShapeLabel_,assEndcapShapeInstanceName_,endcapShapeAssocH);

  //Getting the beamspot from the Event:
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByType(recoBeamSpotHandle);
  const math::XYZPoint bsPosition = recoBeamSpotHandle->position();

  // create electrons 
  const BasicClusterShapeAssociationCollection *shpAssBarrel=&(*barrelShapeAssocH);
  const BasicClusterShapeAssociationCollection *shpAssEndcap=&(*endcapShapeAssocH);
  if (processType_==1) process(tracksH,shpAssBarrel,shpAssEndcap,mhbhe,bsPosition,outEle);
  else {
        edm::Handle<SuperClusterCollection> superClustersBarrelH; 
       e.getByLabel("correctedHybridSuperClusters",superClustersBarrelH);
  
       edm::Handle<SuperClusterCollection> superClustersEndcapH; 
       e.getByLabel("correctedEndcapSuperClustersWithPreshower", superClustersEndcapH);

  process(tracksH, //trackcollection
          superClustersBarrelH, 
          superClustersEndcapH,   
          shpAssBarrel,shpAssEndcap   ,
          mhbhe,  
          bsPosition,
          outEle);
  }

  delete mhbhe;
  std::ostringstream str;

  str << "========== GsfElectronAlgo Info ==========";
  str << "Event " << e.id();
  str << "Number of final electron tracks: " << tracksH.product()->size();
  str << "Number of final electrons: " << outEle.size();
  for (vector<GsfElectron>::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
    str << "New electron with charge, pt, eta, phi : "  << it->charge() << " , " 
        << it->pt() << " , " << it->eta() << " , " << it->phi();
  }
 
  str << "=================================================";
  LogDebug("GsfElectronAlgo") << str.str();
  return;
}
void GsfElectronAlgo::process(edm::Handle<GsfTrackCollection> tracksH,
		        const BasicClusterShapeAssociationCollection *shpAssBarrel,
		        const BasicClusterShapeAssociationCollection *shpAssEndcap,
                        HBHERecHitMetaCollection *mhbhe,
                        const math::XYZPoint &bsPosition,
                        GsfElectronCollection & outEle) {
 
  BasicClusterShapeAssociationCollection::const_iterator seedShpItr;

  const GsfTrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {

    // track -scl association

    const GsfTrack & t=(*tracks)[i];
    const GsfTrackRef trackRef = edm::Ref<GsfTrackCollection>(tracksH,i);
    const SuperClusterRef & scRef=getTrSuperCluster(trackRef);
    const SuperCluster theClus=*scRef;
    std::vector<DetId> vecId=theClus.getHitsByDetId();
    subdet_ =vecId[0].subdetId();  

    //get ref to ClusterShape for seed BasicCluster of SuperCluster
    if (subdet_==EcalEndcap) {
      seedShpItr = shpAssEndcap->find(scRef->seed());
      assert(seedShpItr != shpAssEndcap->end());
    }else if (subdet_==EcalBarrel) {
      seedShpItr = shpAssBarrel->find(scRef->seed());
      assert(seedShpItr != shpAssBarrel->end());
    }
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;

 
    // calculate HoE
    hOverE(scRef,mhbhe);

    // calculate Trajectory StatesOnSurface....
    if (!calculateTSOS(t,theClus,bsPosition)) continue;
    vtxMom_=computeMode(vtxTSOS_);
    sclPos_=sclTSOS_.globalPosition();
    if (preSelection(theClus)) {
      // interface to be improved...
      createElectron(scRef,trackRef ,seedShapeRef, outEle);

      LogInfo("")<<"Constructed new electron with energy  "<< scRef->energy();
    }
  } // loop over tracks
}

bool GsfElectronAlgo::preSelection(const SuperCluster& clus)
{

  LogDebug("")<< "========== preSelection ==========";

  double rt2 = clus.x()*clus.x() + clus.y()*clus.y();
  double r2 = rt2 + clus.z()*clus.z();
  double Et =clus.energy()*sqrt(rt2/r2);

  // pt min
  LogDebug("") << "pT : " << vtxMom_.perp();

  // E/p cut
  LogDebug("") << "E/p : " << clus.energy()/vtxMom_.mag();

  // no E/p preselection for high pT electrons
  if (!highPtPreselection_ || Et <= highPtMin_) {
    if ((subdet_==EcalBarrel) && (clus.energy()/vtxMom_.mag() > maxEOverPBarrel_)) return false;
    if ((subdet_==EcalEndcap) && (clus.energy()/vtxMom_.mag() > maxEOverPEndcaps_)) return false;
    if ((subdet_==EcalBarrel) && (clus.energy()/vtxMom_.mag() < minEOverPBarrel_)) return false;
    if ((subdet_==EcalEndcap) && (clus.energy()/vtxMom_.mag() < minEOverPEndcaps_)) return false;
  }
  LogDebug("") << "E/p criteria is satisfied ";

  // delta eta criteria
  double etaclu = clus.eta();
  double etatrk = sclPos_.eta();
  double deta = etaclu-etatrk;
  LogDebug("") << "delta eta : " << deta;
  if (fabs(deta) > maxDeltaEta_) return false;
  LogDebug("") << "Delta eta criteria is satisfied ";

  // delta phi criteria
  double phiclu = clus.phi();
  double phitrk = sclPos_.phi();
  double dphi = phiclu-phitrk;
  if (fabs(dphi)>CLHEP::pi) dphi = dphi < 0? CLHEP::twopi + dphi : dphi - CLHEP::twopi;
  LogDebug("") << "delta phi : " << dphi;
  if (fabs(dphi) > maxDeltaPhi_) return false;
  LogDebug("") << "Delta phi criteria is satisfied ";

  //H/E cut
  if (HoE_ > maxHOverE_) return false; 
  LogDebug("") << "H/E criteria is satisfied ";

  LogDebug("") << "electron has passed preselection criteria ";
  LogDebug("") << "=================================================";
  return true;  

}  

GlobalVector GsfElectronAlgo::computeMode(const TrajectoryStateOnSurface &tsos) {

  // mode computation for momentum cartesian co-ordinates
  // change to 5D in local parameters??
  float mode_Px = 0.;
  float mode_Py = 0.;
  float mode_Pz = 0.;
  if ( tsos.isValid() ){
    std::vector<TrajectoryStateOnSurface> components(tsos.components());
    unsigned int numb = components.size();
    std::vector<SingleGaussianState1D> pxStates; pxStates.reserve(numb);
    std::vector<SingleGaussianState1D> pyStates; pyStates.reserve(numb);
    std::vector<SingleGaussianState1D> pzStates; pzStates.reserve(numb);
    for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	  ic!=components.end(); ++ic ) {
      GlobalVector momentum(ic->globalMomentum());
      AlgebraicSymMatrix66 cov(ic->cartesianError().matrix());
      pxStates.push_back(SingleGaussianState1D(momentum.x(),cov(3,3),ic->weight()));
      pyStates.push_back(SingleGaussianState1D(momentum.y(),cov(4,4),ic->weight()));
      pzStates.push_back(SingleGaussianState1D(momentum.z(),cov(5,5),ic->weight()));
    }
    MultiGaussianState1D pxState(pxStates);
    MultiGaussianState1D pyState(pyStates);
    MultiGaussianState1D pzState(pzStates);
    GaussianSumUtilities1D pxUtils(pxState);
    GaussianSumUtilities1D pyUtils(pyState);
    GaussianSumUtilities1D pzUtils(pzState);
    mode_Px = pxUtils.mode().mean();
    mode_Py = pyUtils.mode().mean();
    mode_Pz = pzUtils.mode().mean();
  } else edm::LogInfo("") << "tsos not valid!!";
  return GlobalVector(mode_Px,mode_Py,mode_Pz);	

}


 
//FIXME!!
static const float R_ECAL           = 136.5;
static const float Z_Endcap         = 328.0;
static const float etaBarrelEndcap  = 1.479; 

float GsfElectronAlgo::ecalEta(float EtaParticle , float Zvertex, float plane_Radius)
{
  if (EtaParticle!= 0.)
    {
      float Theta = 0.0  ;
      float ZEcal = (R_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;
      
      if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
      if(Theta<0.0) Theta = Theta+Geom::pi() ;

      float ETA = - log(tan(0.5*Theta));
      
      if( fabs(ETA) > etaBarrelEndcap )
	{
	  float Zend = Z_Endcap ;
	  if(EtaParticle<0.0 )  Zend = -Zend ;
	  float Zlen = Zend - Zvertex ;
	  float RR = Zlen/sinh(EtaParticle);
	  Theta = atan((RR+plane_Radius)/Zend);
	  if(Theta<0.0) Theta = Theta+Geom::pi() ;
	  ETA = - log(tan(0.5*Theta));
	}
      return ETA;
    }
  else
    {
      edm::LogWarning("")  << "[EcalPositionFromTrack::etaTransformation] Warning: Eta equals to zero, not correcting" ;
      return EtaParticle;
    }
}

float GsfElectronAlgo::ecalPhi(float PtParticle, float EtaParticle, float PhiParticle, int ChargeParticle, float Rstart)
{
  //Magnetic field
  const float RBARM = 1.357 ;  // was 1.31 , updated on 16122003
  const float ZENDM = 3.186 ;  // was 3.15 , updated on 16122003
  float Rbend = RBARM-(Rstart/100.); //Assumed Rstart in cm
  float Bend  = 0.3 * 4. * Rbend/ 2.0 ;

  //---PHI correction
  float PHI = 0.0 ;
  if( fabs(EtaParticle) <=  etaBarrelEndcap)
    {
      if (fabs(Bend/PtParticle)<=1.)
	{
	  PHI = PhiParticle - asin(Bend/PtParticle)*ChargeParticle;
	  if(PHI >  Geom::pi()) {PHI = PHI - Geom::twoPi();}
	  if(PHI < -Geom::pi()) {PHI = PHI + Geom::twoPi();}
	}
      else
	{
	  edm::LogWarning("") << "[EcalPositionFromTrack::phiTransformation] Warning:Too low Pt, giving up ";
	  return PhiParticle;
	}
    }
  
  if( fabs(EtaParticle) >  etaBarrelEndcap )
    {
      float Rhit = 0.0 ;
      Rhit = ZENDM / sinh(fabs(EtaParticle));
      if (fabs(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)<=1.)
	{
	  PHI = PhiParticle - asin(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)*ChargeParticle;
	  if(PHI >  Geom::pi()) {PHI = PHI - Geom::twoPi();}
	  if(PHI < -Geom::pi()) {PHI = PHI + Geom::twoPi();}
	}
      else
	{
	  edm::LogWarning("") <<"[EcalPositionFromTrack::phiTransformation] Warning:Too low Pt, giving up ";
	  return PhiParticle;
	}
      
    }
  
  //---Return the result
  return PHI;
}

// interface to be improved...
void GsfElectronAlgo::createElectron(const SuperClusterRef & scRef,const GsfTrackRef &trackRef ,const reco::ClusterShapeRef& seedShapeRef, GsfElectronCollection & outEle) {
      GlobalVector innMom=computeMode(innTSOS_);
      GlobalPoint innPos=innTSOS_.globalPosition();
      GlobalVector seedMom=computeMode(seedTSOS_);
      GlobalPoint  seedPos=seedTSOS_.globalPosition();
      GlobalVector sclMom=computeMode(sclTSOS_);    
      GlobalPoint  vtxPos=vtxTSOS_.globalPosition();
      GlobalVector outMom=computeMode(outTSOS_);
      GlobalPoint  outPos=outTSOS_.globalPosition();

      //create electron
      double scale = (*scRef).energy()/vtxMom_.mag();    
      math::XYZTLorentzVectorD momentum= math::XYZTLorentzVector(vtxMom_.x()*scale,
								 vtxMom_.y()*scale,
								 vtxMom_.z()*scale,
								 (*scRef).energy());
     GsfElectron ele(momentum,scRef,seedShapeRef,trackRef,sclPos_,sclMom,seedPos,seedMom,innPos,innMom,vtxPos,vtxMom_,outPos,outMom,HoE_);

      //and set various properties
      float trackEta = ecalEta(
			       trackRef->innerMomentum().eta(),
			       trackRef->innerPosition().z(),
			       trackRef->innerPosition().Rho());

      float trackPhi = ecalPhi(
			       trackRef->innerMomentum().Rho(),
			       trackRef->innerMomentum().eta(),
			       trackRef->innerMomentum().phi(),
			       trackRef->charge(),
			       trackRef->innerPosition().Rho());


      ele.setDeltaEtaSuperClusterAtVtx((*scRef).position().eta() - trackEta);
      float dphi = (*scRef).position().phi() - trackPhi;
      if (fabs(dphi)>CLHEP::pi)
	dphi = dphi < 0? CLHEP::pi2 + dphi : dphi - CLHEP::pi2;
      ele.setDeltaPhiSuperClusterAtVtx(dphi);

      // set corrections + classification
      ElectronClassification theClassifier;
      theClassifier.correct(ele);
      ElectronEnergyCorrector theEnCorrector;
      theEnCorrector.correct(ele, applyEtaCorrection_);
      ElectronMomentumCorrector theMomCorrector;
      theMomCorrector.correct(ele,vtxTSOS_);
      outEle.push_back(ele);
}

void GsfElectronAlgo::hOverE(const SuperClusterRef & scRef,HBHERecHitMetaCollection *mhbhe){
  if (mhbhe) {
    CaloConeSelector sel(hOverEConeSize_, theCaloGeom.product(), DetId::Hcal);
    GlobalPoint pclu((*scRef).x(),(*scRef).y(),(*scRef).z());
    double hcalEnergy = 0.;
    std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,*mhbhe);
    for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
      hcalEnergy += i->energy();
    }
    HoE_= hcalEnergy/(*scRef).energy();
    LogDebug("") << "H/E : " << HoE_;
  } else HoE_=0;

}

const SuperClusterRef GsfElectronAlgo::getTrSuperCluster(const GsfTrackRef & trackRef) {
    edm::RefToBase<TrajectorySeed> seed = trackRef->extra()->seedRef();
    ElectronPixelSeedRef elseed=seed.castTo<ElectronPixelSeedRef>();
    return elseed->superCluster();
}

bool  GsfElectronAlgo::calculateTSOS(const GsfTrack &t,const SuperCluster & theClus,const math::XYZPoint & bsPosition){

    //at innermost point
    innTSOS_ = mtsTransform_->innerStateOnSurface(t, *(trackerHandle_.product()), theMagField.product());
    if (!innTSOS_.isValid()) return false;

    //at vertex
 // innermost state propagation to the beam spot position
    vtxTSOS_
          = TransverseImpactPointExtrapolator(*geomPropBw_).extrapolate(innTSOS_,GlobalPoint(bsPosition.x(),bsPosition.y(),bsPosition.z()));
    if (!vtxTSOS_.isValid()) vtxTSOS_=innTSOS_;

    //at seed
    outTSOS_ 
      = mtsTransform_->outerStateOnSurface(t, *(trackerHandle_.product()), theMagField.product());
    if (!outTSOS_.isValid()) return false;
    
    //    TrajectoryStateOnSurface seedTSOS 
    seedTSOS_ 
     = TransverseImpactPointExtrapolator(*geomPropFw_).extrapolate(outTSOS_,GlobalPoint(theClus.seed()->position().x(),theClus.seed()->position().y(),theClus.seed()->position().z()));
    if (!seedTSOS_.isValid()) seedTSOS_=outTSOS_;

    //at scl
   sclTSOS_ 
    = TransverseImpactPointExtrapolator(*geomPropFw_).extrapolate(innTSOS_,GlobalPoint(theClus.x(),theClus.y(),theClus.z()));
    if (!sclTSOS_.isValid()) sclTSOS_=outTSOS_;
    return true;
}


void GsfElectronAlgo::process(edm::Handle<GsfTrackCollection> tracksH,
                            edm::Handle<reco::SuperClusterCollection> superClustersBarrelH,
                            edm::Handle<reco::SuperClusterCollection> superClustersEndcapH,
                            const reco::BasicClusterShapeAssociationCollection *shpAssBarrel,
	                    const reco::BasicClusterShapeAssociationCollection *shpAssEndcap,
                            HBHERecHitMetaCollection *mhbhe,
                            const math::XYZPoint &bsPosition,  
                            GsfElectronCollection & outEle) {
  
  BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
  //std::cout << "------- processing event" << std::endl;
  
  if (tracksH->size() == 0) {
    //std::cout << "Electron lost: no track found. " << std::endl;
  } else {
    //std::cout << "Number of tracks: " << tracksH->size() << std::endl;
  }
  
  //std::cout << "SuperCluster: " << superClustersBarrelH->size() << "  " << 
  //  superClustersEndcapH->size() << std::endl;	

  reco::SuperClusterRefVector superClusters;

  for(int z=0; z<2; ++z) {

    superClusters.clear();
    if (z == 0) {
      for(reco::SuperClusterCollection::size_type i= 0; i<superClustersBarrelH->size(); ++i){
        reco::SuperClusterRef cluster(superClustersBarrelH, i);
        superClusters.push_back(cluster);
      }
      //std::cout << superClustersBarrelH->size() << std::endl;
    }
    
    if (z == 1) {
      for(reco::SuperClusterCollection::size_type i= 0; i<superClustersEndcapH->size(); ++i){
        reco::SuperClusterRef cluster(superClustersEndcapH, i);
        superClusters.push_back(cluster);
      }
      //std::cout << superClustersEndcapH->size() << std::endl;
    }
    
    //================= loop over SuperClusters ===============

    for(unsigned int i=0; i< superClusters.size(); ++i) {

      //std::cout << "Start matching " << std::endl;	
      reco::SuperClusterRef scRef = superClusters[i];
      reco::SuperCluster theClus = (*scRef);
      reco::GsfTrackRef trackRef = superClusterMatching(scRef, tracksH);
      
      if(trackRef.isNull()) {
        //std::cout << "Electron lost: no supercluster match found: " << tracksH->size() << std::endl;
        continue;
      }
      std::vector<DetId> vecId=theClus.getHitsByDetId();
      subdet_ =vecId[0].subdetId();  

      //get ref to ClusterShape for seed BasicCluster of SuperCluster
      if (subdet_==EcalEndcap) {
	seedShpItr = shpAssEndcap->find(scRef->seed());
	assert(seedShpItr != shpAssEndcap->end());
      }else if (subdet_==EcalBarrel) {
	seedShpItr = shpAssBarrel->find(scRef->seed());
	assert(seedShpItr != shpAssBarrel->end());
      }
      const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
     
      // calculate HoE
      hOverE(scRef,mhbhe);

      // calculate Trajectory StatesOnSurface....
      if (!calculateTSOS((*trackRef),theClus,bsPosition)) continue;

      vtxMom_=computeMode(vtxTSOS_);
      sclPos_=sclTSOS_.globalPosition();
      if (preSelection(theClus)) {
	// interface to be improved...
	createElectron(scRef,trackRef ,seedShapeRef, outEle);
        //LogInfo("")<<"Constructed new electron with energy  "<< (*sclAss)[seed]->energy();
      }
    }  
  }
}

const reco::GsfTrackRef
GsfElectronAlgo::superClusterMatching(reco::SuperClusterRef sc, edm::Handle<reco::GsfTrackCollection> tracks) {

  double minDr = 0.5;
  double minDeop = 10.;
  //reco::SuperClusterRef theClus = edm::Ref<SuperClusterCollection>();
  reco::GsfTrackRef theTrack = edm::Ref<reco::GsfTrackCollection>();


  for(reco::GsfTrackCollection::size_type i=0; i<tracks->size(); ++i){
    reco::GsfTrackRef track(tracks, i);
    math::XYZVector trackGlobalDir(track->momentum());   
    math::XYZVector clusterGlobalDir(sc->x() - track->vx(), sc->y() - track->vy(), sc->z() - track->vz());
    //math::XYZVector clusterGlobalPos(sc->x(), sc->y(), sc->z());
    
    double clusEt = sc->energy()*sin(clusterGlobalDir.theta());
    double clusEstimatedCurvature = clusEt/0.3/4*100;  //4 tesla (temporary solution)
    double DphiBending = sc->position().rho()/2./clusEstimatedCurvature; //ecal radius


    double tmpDr = ROOT::Math::VectorUtil::DeltaR(clusterGlobalDir, trackGlobalDir);
    if ( !(tmpDr < minDr) ) continue;

    TrajectoryStateOnSurface innTSOS = mtsTransform_->innerStateOnSurface(*track, *(trackerHandle_.product()), theMagField.product());
    GlobalVector innMom=computeMode(innTSOS);

    TrajectoryStateOnSurface outTSOS = mtsTransform_->outerStateOnSurface(*track, *(trackerHandle_.product()), theMagField.product());
    if (!outTSOS.isValid())   continue;

    TrajectoryStateOnSurface seedTSOS = TransverseImpactPointExtrapolator(*geomPropFw_).extrapolate(outTSOS,GlobalPoint(sc->seed()->position().x(),sc->seed()->position().y(),sc->seed()->position().z()));
    if (!seedTSOS.isValid()) seedTSOS=outTSOS;

    GlobalVector seedMom=computeMode(seedTSOS);

    double eOverPin  = sc->energy()/innMom.mag();
    //    double eOverPout = sc->seed()->energy()/seedMom.mag();
 
    double Deta = fabs(clusterGlobalDir.eta() - trackGlobalDir.eta());
    double dPhi = fabs(acos(cos(clusterGlobalDir.phi() - trackGlobalDir.phi())));
    float dPhi1 = fabs(dPhi - DphiBending);
    float dPhi2 = fabs(dPhi + DphiBending);

    //    if( !(eOverPout>0.5) ) continue;
    if( !(eOverPin<5) )  continue;
    if( !(min(dPhi1,dPhi2) < 0.1) )  continue;
    if( !(Deta < 0.02) ) continue;

    //    cout << " in matchbox, dphi, deta: " << Dphi << " , " << Deta << endl;
    //    cout << " in matchbox, E/Pin, out: " << eOverPin << " , " << eOverPout << endl;

    if( fabs(eOverPin-1.) < minDeop){
      minDeop = fabs(eOverPin-1.) ;
      theTrack = track;
    }
  }

  //cout << " in matchbox, minD(eop): " << minDeop << endl;
  //std::cout << "returning null ref" << std::endl;
  return theTrack;
}
