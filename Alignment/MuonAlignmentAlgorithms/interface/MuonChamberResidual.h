#ifndef Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H

/** \class MuonChamberResidual
 *  $Date: 2009/04/23 05:06:01 $
 *  $Revision: 1.5 $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class MuonChamberResidual {
public:
  MuonChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : m_globalGeometry(globalGeometry)
    , m_navigator(navigator)
    , m_chamberId(chamberId)
    , m_chamberAlignable(chamberAlignable)
    , m_numHits(0)
    , m_residual_1(0.)
    , m_residual_x(0.)
    , m_residual_y(0.)
    , m_residual_xx(0.)
    , m_residual_xy(0.)
    , m_trackx_1(0.)
    , m_trackx_x(0.)
    , m_trackx_y(0.)
    , m_trackx_xx(0.)
    , m_trackx_xy(0.)
    , m_tracky_1(0.)
    , m_tracky_x(0.)
    , m_tracky_y(0.)
    , m_tracky_xx(0.)
    , m_tracky_xy(0.)
  {};

  virtual ~MuonChamberResidual() {};

  enum {
    kDT13,
    kDT2,
    kCSC
  };

  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) = 0;
  virtual double signConvention(const unsigned int rawId=0) const = 0;

  DetId chamberId() const { return m_chamberId; };
  AlignableDetOrUnitPtr chamberAlignable() const { return m_chamberAlignable; };
  virtual int type() const = 0;

  int numHits() const { return m_numHits; };

  double residual() const {
    assert(m_numHits > 1);
    double delta = m_residual_1*m_residual_xx - m_residual_x*m_residual_x;
    return (m_residual_xx*m_residual_y - m_residual_x*m_residual_xy) / delta;
  };

  double residual_error() const {
    assert(m_numHits > 1);
    double delta = m_residual_1*m_residual_xx - m_residual_x*m_residual_x;
    return sqrt(m_residual_xx / delta);
  };

  double resslope() const {
    assert(m_numHits > 1);
    double delta = m_residual_1*m_residual_xx - m_residual_x*m_residual_x;
    return (m_residual_1*m_residual_xy - m_residual_x*m_residual_y) / delta;
  };

  double resslope_error() const {
    assert(m_numHits > 1);
    double delta = m_residual_1*m_residual_xx - m_residual_x*m_residual_x;
    return sqrt(m_residual_1 / delta);
  };

  double chi2() const {
    double output = 0.;
    double a = residual();
    double b = resslope();

    std::vector<double>::const_iterator x = m_individual_x.begin();
    std::vector<double>::const_iterator y = m_individual_y.begin();
    std::vector<double>::const_iterator w = m_individual_weight.begin();
    for (;  x != m_individual_x.end();  ++x, ++y, ++w) {
      output += pow((*y) - a - b*(*x), 2) * (*w);
    }
    return output;
  };

  int ndof() const {
    return m_individual_x.size() - 2;
  };

  double trackdxdz() const {
    assert(m_numHits > 0);
    double delta = m_trackx_1*m_trackx_xx - m_trackx_x*m_trackx_x;
    return (m_trackx_1*m_trackx_xy - m_trackx_x*m_trackx_y) / delta;
  };

  double trackdydz() const {
    assert(m_numHits > 0);
    double delta = m_tracky_1*m_tracky_xx - m_tracky_x*m_tracky_x;
    return (m_tracky_1*m_tracky_xy - m_tracky_x*m_tracky_y) / delta;
  };

  double trackx() const {
    assert(m_numHits > 0);
    double delta = m_trackx_1*m_trackx_xx - m_trackx_x*m_trackx_x;
    return (m_trackx_xx*m_trackx_y - m_trackx_x*m_trackx_xy) / delta;
  };

  double tracky() const {
    assert(m_numHits > 0);
    double delta = m_tracky_1*m_tracky_xx - m_tracky_x*m_tracky_x;
    return (m_tracky_xx*m_tracky_y - m_tracky_x*m_tracky_xy) / delta;
  };

  GlobalPoint global_trackpos() {
    return chamberAlignable()->surface().toGlobal(LocalPoint(trackx(), tracky(), 0.));
  };

  double hitresid(int i) const {
    assert(0 <= i  &&  i < int(m_localIDs.size()));
    return m_localResids[i];
  }

  double global_residual() const {
    return residual() * signConvention();
  };

  double global_resslope() const {
    return resslope() * signConvention();
  };

  double global_hitresid(int i) const {
    return hitresid(i) * signConvention(m_localIDs[i].rawId());
  };

  int hitlayer(int i) const {  // only difference between DTs and CSCs is the DetId subclass
    assert(0 <= i  &&  i < int(m_localIDs.size()));
    if (m_chamberId.subdetId() == MuonSubdetId::DT) {
      DTLayerId layerId(m_localIDs[i].rawId());
      return 4*(layerId.superlayer() - 1) + layerId.layer();
    }
    else if (m_chamberId.subdetId() == MuonSubdetId::CSC) {
      CSCDetId layerId(m_localIDs[i].rawId());
      return layerId.layer();
    }
    else assert(false);
  };

  double hitposition(int i) const {
    assert(0 <= i  &&  i < int(m_localIDs.size()));
    if (m_chamberId.subdetId() == MuonSubdetId::DT) {
      GlobalPoint pos = m_globalGeometry->idToDet(m_localIDs[i])->position();
      return sqrt(pow(pos.x(), 2) + pow(pos.y(), 2));                   // R for DTs
    }
    else if (m_chamberId.subdetId() == MuonSubdetId::CSC) {
      return m_globalGeometry->idToDet(m_localIDs[i])->position().z();  // Z for CSCs
    }
    else assert(false);
  };

  DetId localid(int i) const {
    return m_localIDs[i];
  };

protected:
  edm::ESHandle<GlobalTrackingGeometry> m_globalGeometry;
  AlignableNavigator *m_navigator;
  DetId m_chamberId;
  AlignableDetOrUnitPtr m_chamberAlignable;

  int m_numHits;
  double m_residual_1;
  double m_residual_x;
  double m_residual_y;
  double m_residual_xx;
  double m_residual_xy;
  double m_trackx_1;
  double m_trackx_x;
  double m_trackx_y;
  double m_trackx_xx;
  double m_trackx_xy;
  double m_tracky_1;
  double m_tracky_x;
  double m_tracky_y;
  double m_tracky_xx;
  double m_tracky_xy;
  std::vector<DetId> m_localIDs;
  std::vector<double> m_localResids;
  std::vector<double> m_individual_x;
  std::vector<double> m_individual_y;
  std::vector<double> m_individual_weight;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
