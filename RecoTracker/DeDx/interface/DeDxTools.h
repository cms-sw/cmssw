#ifndef DeDxTools_H
#define DeDxTools_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"


#include "TFile.h"
#include "TChain.h"
#include "TH3F.h"

namespace DeDxTools  {
 
  struct RawHits {
    double charge;
    double angleCosine;
    DetId detId;
    const TrajectoryMeasurement* trajectoryMeasurement;
    int   NSaturating;
  };

  inline const SiStripCluster* GetCluster(const TrackerSingleRecHit * hit) { return &hit->stripCluster();}
  inline const SiStripCluster* GetCluster(const TrackerSingleRecHit & hit) {return &hit.stripCluster();}
  void   trajectoryRawHits(const edm::Ref<std::vector<Trajectory> >& trajectory, std::vector<RawHits>& hits, bool usePixel, bool useStrip);
  bool shapeSelection(const std::vector<uint8_t> & ampls);

  int getCharge(const SiStripCluster* cluster, int& nSatStrip, const GeomDetUnit& detUnit, const std::vector< std::vector< float > >& calibGains, const unsigned int& m_off );
  void makeCalibrationMap(const std::string& m_calibrationPath, const TrackerGeometry& tkGeom, std::vector< std::vector< float > >& calibGains, const unsigned int& m_off);

  void buildDiscrimMap(edm::Run const& run, const edm::EventSetup& iSetup, std::string Reccord, std::string ProbabilityMode, TH3F*& Prob_ChargePath);

}

#endif
