#ifndef RecoEgamma_EgammaPhotonProducers_ConversionTrackCandidateProducer_h
#define RecoEgamma_EgammaPhotonProducers_ConversionTrackCandidateProducer_h
/** \class ConversionTrackCandidateProducer
 **  
 **
 **  $Id: ConversionTrackCandidateProducer.h,v 1.3 2007/03/26 22:30:12 nancy Exp $ 
 **  $Date: 2007/03/26 22:30:12 $ 
 **  $Revision: 1.3 $
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

#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class OutInConversionSeedFinder;
class InOutConversionSeedFinder;
class OutInConversionTrackFinder;
class InOutConversionTrackFinder;

// ConversionTrackCandidateProducer inherits from EDProducer, so it can be a module:
class ConversionTrackCandidateProducer : public edm::EDProducer {

 public:

  ConversionTrackCandidateProducer (const edm::ParameterSet& ps);
  ~ConversionTrackCandidateProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  int nEvt_;
  

  std::string OutInTrackCandidateCollection_;
  std::string InOutTrackCandidateCollection_;


  std::string OutInTrackSuperClusterAssociationCollection_;
  std::string InOutTrackSuperClusterAssociationCollection_;
  
  std::string bcProducer_;
  std::string bcBarrelCollection_;
  std::string bcEndcapCollection_;
  std::string scHybridBarrelProducer_;
  std::string scIslandEndcapProducer_;
  std::string scHybridBarrelCollection_;
  std::string scIslandEndcapCollection_;
  edm::ParameterSet conf_;


  edm::ESHandle<MagneticField> theMF_;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker_;
 
  const MeasurementTracker*     theMeasurementTracker_;
  const NavigationSchool*       theNavigationSchool_;
  OutInConversionSeedFinder*  theOutInSeedFinder_;
  OutInConversionTrackFinder* theOutInTrackFinder_;
  InOutConversionSeedFinder*  theInOutSeedFinder_;
  InOutConversionTrackFinder* theInOutTrackFinder_;

  const LayerMeasurements*      theLayerMeasurements_;
 
  
  bool isInitialized;


  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForOutIn;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForInOut;  
  
  void buildCollections( const edm::Handle<reco::SuperClusterCollection> & scHandle,
			 const edm::Handle<reco::BasicClusterCollection> & bcHandle,
			 TrackCandidateCollection& outInTracks,
			 TrackCandidateCollection& inOutTracks,
			 std::vector<edm::Ref<reco::SuperClusterCollection> >& vecRecOI,
			 std::vector<edm::Ref<reco::SuperClusterCollection> >& vecRecIO);


};
#endif
