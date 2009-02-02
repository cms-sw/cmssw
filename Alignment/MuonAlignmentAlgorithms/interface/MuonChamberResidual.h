#ifndef Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H

/** \class MuonChamberResidual
 *  $Date: Sat Jan 24 16:32:25 CST 2009 $
 *  $Revision: 1.0 $
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
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class MuonChamberResidual {
public:
  MuonChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
    : m_globalGeometry(globalGeometry)
    , m_navigator(navigator)
    , m_chamberId(chamberId)
    , m_chamberAlignable(chamberAlignable)
  {};

  virtual ~MuonChamberResidual() {};

  AlignableDetOrUnitPtr chamberAlignable() const { return m_chamberAlignable; };
  
  virtual void addResidual(const TrajectoryStateOnSurface *tsos, const TransientTrackingRecHit *hit) = 0;

  virtual bool isRphiValid() = 0;
  virtual bool isZValid() = 0;
  virtual bool isRValid() = 0;
  virtual int rphiHits() = 0;
  virtual int zHits() = 0;
  virtual int rHits() = 0;

  virtual double globalRphiResidual() = 0;
  virtual double globalZResidual() = 0;
  virtual double globalRResidual() = 0;
  virtual double x_residual() = 0;
  virtual double y_residual() = 0;
  virtual double z_residual() = 0;
  virtual double phix_residual() = 0;
  virtual double phiy_residual() = 0;
  virtual double phiz_residual() = 0;

  virtual double phi_position(int which) = 0;
  virtual double z_position(int which) = 0;
  virtual double R_position(int which) = 0;
  virtual double localx_position(int which) = 0;
  virtual double localy_position(int which) = 0;

protected:
  edm::ESHandle<GlobalTrackingGeometry> m_globalGeometry;
  AlignableNavigator *m_navigator;
  DetId m_chamberId;
  AlignableDetOrUnitPtr m_chamberAlignable;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
