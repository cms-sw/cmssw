#ifndef DeDxTools_H
#define DeDxTools_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"

#include "TFile.h"
#include "TChain.h"
#include "TH3F.h"

namespace DeDxTools {
  bool shapeSelection(const SiStripCluster& ampls);
  int getCharge(const SiStripCluster* cluster,
                int& nSatStrip,
                const GeomDetUnit& detUnit,
                const std::vector<std::vector<float>>& calibGains,
                const unsigned int& m_off);
  void makeCalibrationMap(const std::string& m_calibrationPath,
                          const TrackerGeometry& tkGeom,
                          std::vector<std::vector<float>>& calibGains,
                          const unsigned int& m_off);

  using H3DD = PhysicsTools::Calibration::HistogramD3D;
  using ESGetTokenH3DDVariant = std::variant<edm::ESGetToken<H3DD, SiStripDeDxMip_3D_Rcd>,
                                             edm::ESGetToken<H3DD, SiStripDeDxPion_3D_Rcd>,
                                             edm::ESGetToken<H3DD, SiStripDeDxKaon_3D_Rcd>,
                                             edm::ESGetToken<H3DD, SiStripDeDxProton_3D_Rcd>,
                                             edm::ESGetToken<H3DD, SiStripDeDxElectron_3D_Rcd>>;
  ESGetTokenH3DDVariant esConsumes(std::string const& Reccord, edm::ConsumesCollector&);
  PhysicsTools::Calibration::HistogramD3D const& getHistogramD3D(edm::EventSetup const&, ESGetTokenH3DDVariant const&);
  void buildDiscrimMap(PhysicsTools::Calibration::HistogramD3D const&,
                       std::string const& ProbabilityMode,
                       TH3F*& Prob_ChargePath);
  bool IsSpanningOver2APV(unsigned int FirstStrip, unsigned int ClusterSize);
  bool IsFarFromBorder(const TrajectoryStateOnSurface& trajState, const GeomDetUnit* it);
}  // namespace DeDxTools

#endif
