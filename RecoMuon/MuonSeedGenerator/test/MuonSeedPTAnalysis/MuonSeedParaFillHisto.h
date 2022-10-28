#ifndef RecoMuon_MuonSeedParaFillHisto_H
#define RecoMuon_MuonSeedParaFillHisto_H

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

class MuonSeedParaFillHisto {
public:
  /// Constructor
  MuonSeedParaFillHisto();

  /// Destructor
  virtual ~MuonSeedParaFillHisto();

  // Utility functions
  void FillCSCSegmentPair(
      H2DRecHit2* histo2, double pt1[5], double chi2_dof1[5], double dPhiP1[2][5][5], double EtaP1[2][5]);

  void FillDTSegmentPair(
      H2DRecHit3* histo3, double pt1[5], double chi2_dof3[5], double dPhiP3[2][5][5], double EtaP3[2][5]);

  void FillCSCSegmentPairByChamber(H2DRecHit4* hME1[15],
                                   double pt1[5],
                                   double dPhiP1[2][5][5],
                                   double EtaP1[2][5],
                                   bool MEPath[2][5][4],
                                   double dEtaP1[2][5][5]);
  void FillDTSegmentPairByChamber(H2DRecHit5* hMB1[26],
                                  double pt1[5],
                                  double dPhiP3[2][5][5],
                                  double EtaP3[2][5],
                                  bool MBPath[2][5][3],
                                  double dEtaP3[2][5][5]);

  void FillCSCSegmentSingle(
      H2DRecHit6* hME2[8], double pt1[5], double ME_phi[2][5][4], double ME_eta[2][5][4], bool MEPath[2][5][4]);

  void FillDTSegmentSingle(
      H2DRecHit7* hMB2[12], double pt1[5], double MB_phi[2][5][3], double MB_eta[2][5][3], bool MBPath[2][5][3]);

  void FillOLSegmentPairByChamber(H2DRecHit10* hOL1[6],
                                  double pt1[5],
                                  double dPhiP2[2][5][5],
                                  double EtaP3[2][5],
                                  bool MBPath[2][5][3],
                                  bool MEPath[2][5][4],
                                  double dEtaP2[2][5][5]);

protected:
private:
  // dPhi and dEta for CSC
  /*
  double PhiV1[2][5];
  double EtaV1[2][5];
  double dPhiV1[2][5][5];
  double dEtaV1[2][5][5];
  double PhiP1[2][5];
  double EtaP1[2][5];
  double dPhiP1[2][5][5];
  double dEtaP1[2][5][5];
  double chi2_dof1[5];
  /// dphi and eta for CSC single segment
  bool   MEPath[2][5][4];
  double ME_phi[2][5][4];
  double ME_eta[2][5][4];

  // dPhi and dEta for DT
  double PhiV3[2][5];
  double EtaV3[2][5];
  double dPhiV3[2][5][5];
  double dEtaV3[2][5][5];
  double PhiP3[2][5];
  double EtaP3[2][5];
  double dPhiP3[2][5][5];
  double dEtaP3[2][5][5];
  double chi2_dof3[5];
  /// dphi and eta for DT single segment
  bool   MBPath[2][5][3];
  double MB_phi[2][5][3];
  double MB_eta[2][5][3];

  // dphi & Eta for Overlap region
  double dPhiV2[2][5][5];
  double dEtaV2[2][5][5];
  double dPhiP2[2][5][5];
  double dEtaP2[2][5][5];
  */

  //std::string rootFileName;
};

#endif
