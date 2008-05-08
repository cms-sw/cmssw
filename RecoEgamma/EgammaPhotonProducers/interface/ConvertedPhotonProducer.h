#ifndef RecoEgamma_EgammaPhotonProducers_ConvertedPhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_ConvertedPhotonProducer_h
/** \class ConvertedPhotonProducer
 **  
 **
 **  $Id: ConvertedPhotonProducer.h,v 1.16 2008/02/15 16:51:21 nancy Exp $ 
 **  $Date: 2008/02/15 16:51:21 $ 
 **  $Revision: 1.16 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"

class ConversionTrackEcalImpactPoint;
class ConversionTrackPairFinder;
class ConversionVertexFinder;
class ConvertedPhotonProducer : public edm::EDProducer {

 public:

  ConvertedPhotonProducer (const edm::ParameterSet& ps);
  ~ConvertedPhotonProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void endJob ();
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void buildCollections ( const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
			  const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
			  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>& allPairs,
			  reco::ConversionCollection & outputConvPhotonCollection);
    
  
  std::string conversionOITrackProducer_;
  std::string conversionIOTrackProducer_;


  std::string outInTrackSCAssociationCollection_;
  std::string inOutTrackSCAssociationCollection_;


  std::string ConvertedPhotonCollection_;
  
  std::string bcProducer_;
  std::string bcBarrelCollection_;
  std::string bcEndcapCollection_;
  std::string scHybridBarrelProducer_;
  std::string scIslandEndcapProducer_;
  std::string scHybridBarrelCollection_;
  std::string scIslandEndcapCollection_;
  edm::ParameterSet conf_;

  edm::ESHandle<MagneticField> theMF_;

  ConversionTrackPairFinder*      theTrackPairFinder_;
  ConversionVertexFinder*         theVertexFinder_;
  ConversionTrackEcalImpactPoint* theEcalImpactPositionFinder_;


  int nEvt_;


};
#endif
