#ifndef RecoEgamma_EgammaPhotonProducers_SoftConversionTrackCandidateProducer_h
#define RecoEgamma_EgammaPhotonProducers_SoftConversionTrackCandidateProducer_h
/** \class SoftConversionTrackCandidateProducer
 **  
 **
 **  $Id: SoftConversionTrackCandidateProducer.h,v 1.1 2008/05/28 03:45:36 dwjang Exp $ 
 **  $Date: 2008/05/28 03:45:36 $ 
 **  $Revision: 1.1 $
 **  \author Dongwook Jang, Carnegie Mellon University
 **  Modified version of original ConversionTrackCandidateProducer
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"

class OutInConversionSeedFinder;
class InOutConversionSeedFinder;
class OutInConversionTrackFinder;
class InOutConversionTrackFinder;

// SoftConversionTrackCandidateProducer inherits from EDProducer, so it can be a module:
class SoftConversionTrackCandidateProducer : public edm::EDProducer {

 public:

  SoftConversionTrackCandidateProducer (const edm::ParameterSet& ps);
  ~SoftConversionTrackCandidateProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;
  
 /// Initialize EventSetup objects at each event
  void setEventSetup( const edm::EventSetup& es ) ;

  std::string clusterType_;
  std::string clusterProducer_;
  std::string clusterBarrelCollection_;
  std::string clusterEndcapCollection_;
  std::string OutInTrackCandidateCollection_;
  std::string InOutTrackCandidateCollection_;
  std::string OutInTrackClusterAssociationCollection_;
  std::string InOutTrackClusterAssociationCollection_;
  
  edm::ParameterSet conf_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;  

  const NavigationSchool*     theNavigationSchool_;
  OutInConversionSeedFinder*  theOutInSeedFinder_;
  OutInConversionTrackFinder* theOutInTrackFinder_;
  InOutConversionSeedFinder*  theInOutSeedFinder_;
  InOutConversionTrackFinder* theInOutTrackFinder_;

  void buildCollections( const edm::Handle<edm::View<reco::CaloCluster> >& clusterHandle,
			 TrackCandidateCollection& outInTracks,
			 TrackCandidateCollection& inOutTracks,
			 std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
			 std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO);

};
#endif
