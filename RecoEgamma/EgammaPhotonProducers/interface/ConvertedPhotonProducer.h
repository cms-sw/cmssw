#ifndef RecoEgamma_EgammaPhotonProducers_ConvertedPhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_ConvertedPhotonProducer_h
/** \class ConvertedPhotonProducer
 **  
 **
 **  $Id: ConvertedPhotonProducer.h,v 1.32 2010/01/15 18:25:53 nancy Exp $ 
 **  $Date: 2010/01/15 18:25:53 $ 
 **  $Revision: 1.32 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class ConversionTrackEcalImpactPoint;
class ConversionTrackPairFinder;
class ConversionVertexFinder;
class ConvertedPhotonProducer : public edm::EDProducer {

 public:

  ConvertedPhotonProducer (const edm::ParameterSet& ps);
  ~ConvertedPhotonProducer();

  virtual void beginRun (edm::Run& r, edm::EventSetup const & es);
  virtual void endRun (edm::Run& r, edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);
  virtual void endJob();


 private:
  





  void buildCollections ( edm::EventSetup const & es,
                          const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
			  const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
			  const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
			  const edm::Handle<reco::TrackCollection>  & trkHandle,
			  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>& allPairs,
			  reco::ConversionCollection & outputConvPhotonCollection);
  void cleanCollections (const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
			 const edm::OrphanHandle<reco::ConversionCollection> & conversionHandle,
			 reco::ConversionCollection & outputCollection);
			   
  std::vector<reco::ConversionRef> solveAmbiguity( const edm::OrphanHandle<reco::ConversionCollection> & conversionHandle, reco::CaloClusterPtr& sc);

  float calculateMinApproachDistance ( const reco::TrackRef& track1, const reco::TrackRef& track2);
  void getCircleCenter(const reco::TrackRef& tk, double r, double& x0, double& y0);
    
  
  std::string conversionOITrackProducer_;
  std::string conversionIOTrackProducer_;


  std::string outInTrackSCAssociationCollection_;
  std::string inOutTrackSCAssociationCollection_;


  std::string ConvertedPhotonCollection_;
  std::string CleanedConvertedPhotonCollection_;
  
  edm::InputTag bcBarrelCollection_;
  edm::InputTag bcEndcapCollection_;
  edm::InputTag scHybridBarrelProducer_;
  edm::InputTag scIslandEndcapProducer_;
  edm::ParameterSet conf_;
  edm::InputTag hcalTowers_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<MagneticField> theMF_;
  edm::ESHandle<TransientTrackBuilder> theTransientTrackBuilder_;

  ConversionTrackPairFinder*      theTrackPairFinder_;
  ConversionVertexFinder*         theVertexFinder_;
  ConversionTrackEcalImpactPoint* theEcalImpactPositionFinder_;
  int nEvt_;
  std::string algoName_;

 
  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  bool  recoverOneTrackCase_;
  double dRForConversionRecovery_;
  double deltaCotCut_;
  double minApproachDisCut_;
  int    maxNumOfCandidates_;
  bool risolveAmbiguity_;


  ConversionLikelihoodCalculator* theLikelihoodCalc_;
  std::string likelihoodWeights_;



};
#endif
