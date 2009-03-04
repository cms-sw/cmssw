#ifndef TrackRecoDeDx_DeDxEstimatorProducer_H
#define TrackRecoDeDx_DeDxEstimatorProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 



//
// class declaration
//

class DeDxEstimatorProducer : public edm::EDProducer {

public:

  explicit DeDxEstimatorProducer(const edm::ParameterSet&);
  ~DeDxEstimatorProducer();

private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  double thickness    (DetId id);
  double normalization(DetId id);
  double distance     (DetId id);


  // ----------member data ---------------------------
  BaseDeDxEstimator*                m_estimator;

  edm::InputTag                     m_trajTrackAssociationTag;
  edm::InputTag                     m_tracksTag;

  bool usePixel;
  bool useStrip;
  double MeVperADCPixel;
  double MeVperADCStrip;

  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;

  const TrackerGeometry* m_tracker;
  std::map<DetId,double> m_normalizationMap;
  std::map<DetId,double> m_distanceMap;
  std::map<DetId,double> m_thicknessMap;
};

#endif

