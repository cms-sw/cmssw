#ifndef TrackRecoDeDx_DeDxDiscriminatorDumpFromDB_H
#define TrackRecoDeDx_DeDxDiscriminatorDumpFromDB_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorTools.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"


#include "TH3F.h"
#include "TChain.h"


class DeDxDiscriminatorDumpFromDB : public edm::EDProducer {

public:

  explicit DeDxDiscriminatorDumpFromDB(const edm::ParameterSet&);
  ~DeDxDiscriminatorDumpFromDB();

private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  PhysicsTools::Calibration::HistogramD3D DeDxMap_;
  std::string       Reccord;
  std::string       HistoFile;
};

#endif

