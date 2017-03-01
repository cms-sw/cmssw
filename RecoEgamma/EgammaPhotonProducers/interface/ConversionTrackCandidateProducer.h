#ifndef RecoEgamma_EgammaPhotonProducers_ConversionTrackCandidateProducer_h
#define RecoEgamma_EgammaPhotonProducers_ConversionTrackCandidateProducer_h
/** \class ConversionTrackCandidateProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

class OutInConversionSeedFinder;
class InOutConversionSeedFinder;
class OutInConversionTrackFinder;
class InOutConversionTrackFinder;

// ConversionTrackCandidateProducer inherits from EDProducer, so it can be a module:
class ConversionTrackCandidateProducer : public edm::stream::EDProducer<> {

 public:

  ConversionTrackCandidateProducer (const edm::ParameterSet& ps);
  ~ConversionTrackCandidateProducer();
  
  virtual void beginRun (edm::Run const&, edm::EventSetup const & es) override final;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;

 private:

  int nEvt_;
  
 /// Initialize EventSetup objects at each event
  void setEventSetup( const edm::EventSetup& es ) ;

  std::string OutInTrackCandidateCollection_;
  std::string InOutTrackCandidateCollection_;


  std::string OutInTrackSuperClusterAssociationCollection_;
  std::string InOutTrackSuperClusterAssociationCollection_;
  
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > bcBarrelCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > bcEndcapCollection_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > scHybridBarrelProducer_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > scIslandEndcapProducer_;
  edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelecalCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapecalCollection_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrkEvtToken_;
 
  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double isoConeR_   ;   
  double isoInnerConeR_ ;
  double isoEtaSlice_   ;
  double isoEtMin_      ;
  double isoEMin_       ;
  bool   vetoClusteredHits_ ;
  bool   useNumXtals_;

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double ecalIsoCut_offset_;
  double ecalIsoCut_slope_;


  edm::ESHandle<CaloGeometry> theCaloGeom_;  

  std::unique_ptr<BaseCkfTrajectoryBuilder> theTrajectoryBuilder_;

  std::unique_ptr<OutInConversionSeedFinder>  theOutInSeedFinder_;
  std::unique_ptr<OutInConversionTrackFinder> theOutInTrackFinder_;
  std::unique_ptr<InOutConversionSeedFinder>  theInOutSeedFinder_;
  std::unique_ptr<InOutConversionTrackFinder> theInOutTrackFinder_;


  std::vector<edm::Ptr<reco::CaloCluster> > caloPtrVecOutIn_; 
  std::vector<edm::Ptr<reco::CaloCluster> > caloPtrVecInOut_; 

  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForOutIn;  
  std::vector<edm::Ref<reco::SuperClusterCollection> > vecOfSCRefForInOut;  
  
  void buildCollections(bool detector, 
			const edm::Handle<edm::View<reco::CaloCluster> > & scHandle,
			const edm::Handle<edm::View<reco::CaloCluster> > & bcHandle,
			edm::Handle<EcalRecHitCollection> ecalRecHitHandle, 
			const EcalRecHitCollection& ecalRecHits,
			const EcalSeverityLevelAlgo* sevLev,
			//edm::ESHandle<EcalChannelStatus>  chStatus,
			const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
			TrackCandidateCollection& outInTracks,
			TrackCandidateCollection& inOutTracks,
			std::vector<edm::Ptr<reco::CaloCluster> >& vecRecOI,
			std::vector<edm::Ptr<reco::CaloCluster> >& vecRecIO);

};
#endif
