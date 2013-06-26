#ifndef TrackRecoDeDx_DeDxEstimatorProducer_H
#define TrackRecoDeDx_DeDxEstimatorProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include <ext/hash_map>

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "TFile.h"
#include "TChain.h"


//
// class declaration
//

class DeDxEstimatorProducer : public edm::EDProducer {

public:

  explicit DeDxEstimatorProducer(const edm::ParameterSet&);
  ~DeDxEstimatorProducer();

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

  int    getCharge(const SiStripCluster*   Cluster, int& Saturating_Strips,const uint32_t &);
//  int    getCharge(const SiStripRecHit2D* sistripsimplehit, int& Saturating_Strips);
  void   MakeCalibrationMap();


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

  std::string                       m_calibrationPath;
  bool                              useCalibration;
  bool                              shapetest;

   private : 
      struct stModInfo{int DetId; float Thickness; float Distance; float Normalization; double Gain;};

      class isEqual{
         public:
                 template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
      };
  
  __gnu_cxx::hash_map<unsigned int, stModInfo*,  __gnu_cxx::hash<unsigned int>, isEqual > MODsColl;
};

#endif

