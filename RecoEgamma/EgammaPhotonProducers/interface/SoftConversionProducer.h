#ifndef RecoEgamma_EgammaPhotonProducers_SoftConversionProducer_h
#define RecoEgamma_EgammaPhotonProducers_SoftConversionProducer_h
/** \class SoftConversionProducer
 **  
 **
 **  $Id: SoftConversionProducer.h,v 1.1 2008/05/28 03:45:36 dwjang Exp $ 
 **  $Date: 2008/05/28 03:45:36 $ 
 **  $Revision: 1.1 $
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

class ConversionTrackEcalImpactPoint;
class ConversionTrackPairFinder;
class ConversionVertexFinder;
class SoftConversionProducer : public edm::EDProducer {

 public:

  typedef std::vector<std::pair<reco::TrackRef, reco::CaloClusterPtr> > TrackClusterMap;

  SoftConversionProducer (const edm::ParameterSet& ps);
  ~SoftConversionProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void endJob ();
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  std::string conversionOITrackProducer_;
  std::string conversionIOTrackProducer_;

  std::string outInTrackClusterAssociationCollection_;
  std::string inOutTrackClusterAssociationCollection_;

  std::string clusterType_;
  std::string clusterProducer_;
  std::string clusterBarrelCollection_;
  std::string clusterEndcapCollection_;

  std::string softConversionCollection_;
  
  edm::ParameterSet conf_;
  edm::ESHandle<MagneticField> theMF_;
  edm::ESHandle<GeometricSearchTracker>       theGeomSearchTracker_;
 
  ConversionTrackPairFinder*      theTrackPairFinder_;
  ConversionVertexFinder*         theVertexFinder_;
  ConversionTrackEcalImpactPoint* theEcalImpactPositionFinder_;

};
#endif
