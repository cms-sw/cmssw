#ifndef Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H

/** \class MuonChamberResidual
 * 
 * Abstract base class for muon chamber residulas
 * 
 *  $Id: MuonChamberResidual.h,v 1.7 2011/10/12 23:40:24 khotilov Exp $
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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class MuonChamberResidual
{
public:

  enum {
    kDT13,
    kDT2,
    kCSC
  };

  MuonChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator, 
                      DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);

  virtual ~MuonChamberResidual() {}

  // has to be implemented for rechit based residuals 
  virtual void addResidual(const TrajectoryStateOnSurface *, const TransientTrackingRecHit *) = 0;
  
  // has to be implemented for track muon segment residuals
  virtual void setSegmentResidual(const reco::MuonChamberMatch *, const reco::MuonSegmentMatch *) = 0;
  
  int type() const { return m_type; }

  virtual double signConvention() const {return m_sign; }

  DetId chamberId() const { return m_chamberId; }
  
  AlignableDetOrUnitPtr chamberAlignable() const { return m_chamberAlignable; }

  int numHits() const { return m_numHits; }

  double residual() const { return m_residual; }
  double residual_error() const { return m_residual_error; }
  double resslope() const { return m_resslope; }
  double resslope_error() const { return m_resslope_error; }

  double chi2() const { return m_chi2; }
  int ndof() const { return m_ndof; }

  double trackdxdz() const { return m_trackdxdz; }
  double trackdydz() const { return m_trackdydz; }
  double trackx() const { return m_trackx; }
  double tracky() const { return m_tracky; }

  double segdxdz() const { return m_segdxdz; }
  double segdydz() const { return m_segdydz; }
  double segx() const { return m_segx; }
  double segy() const { return m_segy; }

  align::GlobalPoint global_trackpos();
  align::GlobalPoint global_stubpos();
  double global_residual() const;
  double global_resslope() const;
  double global_hitresid(int i) const;
  
  // individual hit methods
  double hitresid(int i) const;
  int hitlayer(int i) const;
  double hitposition(int i) const;
  DetId localid(int i) const { return m_localIDs[i];  }

protected:
  edm::ESHandle<GlobalTrackingGeometry> m_globalGeometry;
  AlignableNavigator *m_navigator;
  DetId m_chamberId;
  AlignableDetOrUnitPtr m_chamberAlignable;

  int m_numHits;
  std::vector<DetId> m_localIDs;
  std::vector<double> m_localResids;
  std::vector<double> m_individual_x;
  std::vector<double> m_individual_y;
  std::vector<double> m_individual_weight;

  int m_type;
  double m_sign;
  double m_chi2;
  int m_ndof;
  double m_residual;
  double m_residual_error;
  double m_resslope;
  double m_resslope_error;
  double m_trackdxdz;
  double m_trackdydz;
  double m_trackx;
  double m_tracky;
  double m_segdxdz;
  double m_segdydz;
  double m_segx;
  double m_segy;
  
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonChamberResidual_H
