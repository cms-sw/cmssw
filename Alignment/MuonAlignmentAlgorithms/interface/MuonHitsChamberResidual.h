#ifndef Alignment_MuonAlignmentAlgorithms_MuonHitsChamberResidual_H
#define Alignment_MuonAlignmentAlgorithms_MuonHitsChamberResidual_H

/** \class MuonHitsChamberResidual
 * 
 * Second level abstraction class for muon chamber residulas: 
 * for alignment using individual rechits it implements linear segment fit of hits.
 * 
 *  $Id: MuonHitsChamberResidual.h,v 1.1 2011/10/12 23:32:07 khotilov Exp $
 *  \author V. Khotilovich - Texas A&M University <khotilov@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"

class MuonHitsChamberResidual : public MuonChamberResidual
{
public:
  
  MuonHitsChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                          DetId chamberId, AlignableDetOrUnitPtr chamberAlignable);
  
  void segment_fit();
  
protected:

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
  double m_hitx_1;
  double m_hitx_x;
  double m_hitx_y;
  double m_hitx_xx;
  double m_hitx_xy;
  double m_hity_1;
  double m_hity_x;
  double m_hity_y;
  double m_hity_xx;
  double m_hity_xy;
};

#endif // Alignment_MuonAlignmentAlgorithms_MuonHitsChamberResidual_H
