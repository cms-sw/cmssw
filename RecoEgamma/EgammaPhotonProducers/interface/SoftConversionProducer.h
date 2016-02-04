#ifndef RecoEgamma_EgammaPhotonProducers_SoftConversionProducer_h
#define RecoEgamma_EgammaPhotonProducers_SoftConversionProducer_h
/** \class SoftConversionProducer
 **  
 **
 **  $Id: SoftConversionProducer.h,v 1.8 2009/03/04 21:20:14 nancy Exp $ 
 **  $Date: 2009/03/04 21:20:14 $ 
 **  $Revision: 1.8 $
 **  \author Dongwook Jang, Carnegie Mellon University
 **  Modified version of ConvertedPhotonProducer
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class ConversionTrackEcalImpactPoint;
class ConversionTrackPairFinder;
class ConversionVertexFinder;
class SoftConversionProducer : public edm::EDProducer {

 public:

  typedef std::vector<std::pair<reco::TrackRef, reco::CaloClusterPtr> > TrackClusterMap;

  SoftConversionProducer (const edm::ParameterSet& ps);
  ~SoftConversionProducer();


  virtual void beginRun (edm::Run& r, edm::EventSetup const & es);
  virtual void endRun(edm::Run &,  edm::EventSetup const&);
    
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);
  bool trackQualityCut(const reco::TrackRef& trk);
  bool NotAlreadyIn(const reco::Conversion& thisConv,
		    const std::auto_ptr<reco::ConversionCollection>& outputColl) const;

 private:

  std::string conversionOITrackProducer_;
  std::string conversionIOTrackProducer_;

  std::string outInTrackClusterAssociationCollection_;
  std::string inOutTrackClusterAssociationCollection_;

  std::string clusterType_;
  edm::InputTag clusterBarrelCollection_;
  edm::InputTag clusterEndcapCollection_;

  std::string softConversionCollection_;
  double trackMaxChi2_;
  double trackMinHits_;
  double clustersMaxDeltaEta_;
  double clustersMaxDeltaPhi_;
  
  edm::ParameterSet conf_;
  edm::ESHandle<MagneticField> theMF_;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker_;
 
  ConversionTrackPairFinder*      theTrackPairFinder_;
  ConversionVertexFinder*         theVertexFinder_;
  ConversionTrackEcalImpactPoint* theEcalImpactPositionFinder_;

};
#endif
