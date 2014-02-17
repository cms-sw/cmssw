#ifndef Alignment_MuonAlignmentAlgorithms_CSCPairResidualsConstraint_H
#define Alignment_MuonAlignmentAlgorithms_CSCPairResidualsConstraint_H

/** \class CSCPairResidualsConstraint
 *  $Date: 2010/05/27 19:40:03 $
 *  $Revision: 1.1 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include <fstream>

#include "TH1F.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/CSCPairConstraint.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

class CSCOverlapsAlignmentAlgorithm;

class CSCPairResidualsConstraint : public CSCPairConstraint {
public:
  CSCPairResidualsConstraint(unsigned int identifier, int i, int j, CSCDetId id_i, CSCDetId id_j)
    : CSCPairConstraint(i, j, 0., 0.)
    , m_identifier(identifier), m_id_i(id_i), m_id_j(id_j)
    , m_sum1(0.), m_sumx(0.), m_sumy(0.), m_sumxx(0.), m_sumyy(0.), m_sumxy(0.), m_sumN(0)
    , m_Zplane(1000.), m_iZ1(1000.), m_iZ6(1000.), m_jZ1(1000.), m_jZ6(1000.), m_cscGeometry(NULL), m_propagator(NULL)
  {};
  virtual ~CSCPairResidualsConstraint() {};

  enum {
    kModePhiy,
    kModePhiPos,
    kModePhiz,
    kModeRadius
  };

  double value() const;
  double error() const;
  CSCDetId id_i() const { return m_id_i; };
  CSCDetId id_j() const { return m_id_j; };
  bool valid() const;
  double radius(bool is_i) const { return m_cscGeometry->idToDet((is_i ? m_id_i : m_id_j))->surface().position().perp(); };

  void configure(CSCOverlapsAlignmentAlgorithm *parent);
  void setZplane(const CSCGeometry *cscGeometry);
  void setPropagator(const Propagator *propagator);
  bool addTrack(const std::vector<TrajectoryMeasurement> &measurements, const reco::TransientTrack &track, const TrackTransformer *trackTransformer);

  void write(std::ofstream &output);
  void read(std::vector<std::ifstream*> &input, std::vector<std::string> &filenames);

protected:
  void calculatePhi(const TransientTrackingRecHit *hit, double &phi, double &phierr2, bool doRphi=false, bool globalPhi=false);
  bool isFiducial(std::vector<const TransientTrackingRecHit*> &hits, bool is_i);
  bool dphidzFromTrack(const std::vector<TrajectoryMeasurement> &measurements, const reco::TransientTrack &track, const TrackTransformer *trackTransformer, double &drphidz);

  unsigned int m_identifier;
  CSCDetId m_id_i, m_id_j;
  double m_sum1, m_sumx, m_sumy, m_sumxx, m_sumyy, m_sumxy;
  int m_sumN;

  CSCOverlapsAlignmentAlgorithm *m_parent;

  double m_Zplane, m_iZ, m_jZ, m_iZ1, m_iZ6, m_jZ1, m_jZ6, m_averageRadius;
  const CSCGeometry *m_cscGeometry;
  const Propagator *m_propagator;
  Plane::PlanePointer m_Zsurface;

  TH1F *m_slopeResiduals;
  TH1F *m_offsetResiduals;
  TH1F *m_radial;
};

#endif // Alignment_MuonAlignmentAlgorithms_CSCPairResidualsConstraint_H
