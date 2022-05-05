#ifndef RecoMuon_MuonSeeddPhiScale_H
#define RecoMuon_MuonSeeddPhiScale_H

/** \class SeedParametrization
 *
 *  Author: S.C. Kao  - UC Riverside
 */

#include "MuonSeedParameterHisto.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment2D.h>
#include <DataFormats/DTRecHit/interface/DTChamberRecSegment2D.h>
#include <DataFormats/DTRecHit/interface/DTRecHitCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecHit1D.h>

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/DTGeometry/interface/DTChamber.h>
#include <Geometry/DTGeometry/interface/DTLayer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>

#include <vector>
#include <map>
#include <string>
#include <utility>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//class PSimHit;

class MuonSeeddPhiScale {
public:
  /// Constructor
  MuonSeeddPhiScale(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~MuonSeeddPhiScale();

  // Utility functions
  void ScaleCSCdPhi(double dPhiP1[2][5][5], double EtaP1[2][5]);

  void ScaleDTdPhi(double dPhiP3[2][5][5], double EtaP3[2][5]);

  void ScaleOLdPhi(double dPhiP2[2][5][5], bool MBPath[2][5][3], bool MEPath[2][5][4]);

  void ScaleMESingle(double ME_phi[2][5][4], bool MEPath[2][5][4]);
  void ScaleMBSingle(double MB_phi[2][5][3], bool MBPath[2][5][3]);

protected:
private:
  std::vector<double> CSC01_1;
  std::vector<double> CSC12_1;
  std::vector<double> CSC12_2;
  std::vector<double> CSC12_3;
  std::vector<double> CSC13_2;
  std::vector<double> CSC13_3;
  std::vector<double> CSC14_3;
  std::vector<double> CSC23_1;
  std::vector<double> CSC23_2;
  std::vector<double> CSC24_1;
  std::vector<double> CSC34_1;

  std::vector<double> DT12_1;
  std::vector<double> DT12_2;
  std::vector<double> DT13_1;
  std::vector<double> DT13_2;
  std::vector<double> DT14_1;
  std::vector<double> DT14_2;
  std::vector<double> DT23_1;
  std::vector<double> DT23_2;
  std::vector<double> DT24_1;
  std::vector<double> DT24_2;
  std::vector<double> DT34_1;
  std::vector<double> DT34_2;

  std::vector<double> OL1213;
  std::vector<double> OL1222;
  std::vector<double> OL1232;
  std::vector<double> OL2213;
  std::vector<double> OL2222;

  std::vector<double> SMB_10S;
  std::vector<double> SMB_11S;
  std::vector<double> SMB_12S;
  std::vector<double> SMB_20S;
  std::vector<double> SMB_21S;
  std::vector<double> SMB_22S;
  std::vector<double> SMB_30S;
  std::vector<double> SMB_31S;
  std::vector<double> SMB_32S;

  std::vector<double> SME_11S;
  std::vector<double> SME_12S;
  std::vector<double> SME_13S;
  std::vector<double> SME_21S;
  std::vector<double> SME_22S;
};

#endif
