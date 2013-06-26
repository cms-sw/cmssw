/** \class MuonHitsChamberResidual
 *  $Id: MuonHitsChamberResidual.cc,v 1.1 2011/10/12 23:32:08 khotilov Exp $
 *  \author V. Khotilovich - Texas A&M University <khotilov@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonHitsChamberResidual.h"

MuonHitsChamberResidual::MuonHitsChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                                 AlignableNavigator *navigator, 
                                                 DetId chamberId,
                                                 AlignableDetOrUnitPtr chamberAlignable)
  : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
  , m_residual_1(0.)
  , m_residual_x(0.)  , m_residual_y(0.)
  , m_residual_xx(0.) , m_residual_xy(0.)
  , m_trackx_1(0.)
  , m_trackx_x(0.)  , m_trackx_y(0.)
  , m_trackx_xx(0.) , m_trackx_xy(0.)
  , m_tracky_1(0.)
  , m_tracky_x(0.)  , m_tracky_y(0.)
  , m_tracky_xx(0.) , m_tracky_xy(0.)
  , m_hitx_1(0.)
  , m_hitx_x(0.)  , m_hitx_y(0.)
  , m_hitx_xx(0.) , m_hitx_xy(0.)
  , m_hity_1(0.)
  , m_hity_x(0.)  , m_hity_y(0.)
  , m_hity_xx(0.) , m_hity_xy(0.)
{}


void MuonHitsChamberResidual::segment_fit()
{
  assert(m_numHits > 1);
  
  double delta = m_residual_1 * m_residual_xx - m_residual_x * m_residual_x;
  m_residual = (m_residual_xx * m_residual_y - m_residual_x * m_residual_xy) / delta;
  
  delta = m_residual_1 * m_residual_xx - m_residual_x * m_residual_x;
  m_residual_error = sqrt(m_residual_xx / delta);
  
  delta = m_residual_1 * m_residual_xx - m_residual_x * m_residual_x;
  m_resslope = (m_residual_1 * m_residual_xy - m_residual_x * m_residual_y) / delta;
  
  delta = m_residual_1 * m_residual_xx - m_residual_x * m_residual_x;
  m_resslope_error = sqrt(m_residual_1 / delta);
  
  m_ndof = m_individual_x.size() - 2;
  
  m_chi2 = 0.;
  double a = m_residual;
  double b = m_resslope;
  std::vector< double >::const_iterator x = m_individual_x.begin();
  std::vector< double >::const_iterator y = m_individual_y.begin();
  std::vector< double >::const_iterator w = m_individual_weight.begin();
  for (; x != m_individual_x.end(); ++x, ++y, ++w)   m_chi2 += pow((*y) - a - b * (*x), 2) * (*w);

  delta = m_trackx_1 * m_trackx_xx - m_trackx_x * m_trackx_x;
  m_trackdxdz = (m_trackx_1 * m_trackx_xy - m_trackx_x * m_trackx_y) / delta;
  
  delta = m_tracky_1 * m_tracky_xx - m_tracky_x * m_tracky_x;
  m_trackdydz = (m_tracky_1 * m_tracky_xy - m_tracky_x * m_tracky_y) / delta;
  
  delta = m_trackx_1 * m_trackx_xx - m_trackx_x * m_trackx_x;
  m_trackx = (m_trackx_xx * m_trackx_y - m_trackx_x * m_trackx_xy) / delta;
  
  delta = m_tracky_1 * m_tracky_xx - m_tracky_x * m_tracky_x;
  m_tracky = (m_tracky_xx * m_tracky_y - m_tracky_x * m_tracky_xy) / delta;

  delta = m_hitx_1 * m_hitx_xx - m_hitx_x * m_hitx_x;
  m_segdxdz = (m_hitx_1 * m_hitx_xy - m_hitx_x * m_hitx_y) / delta;

  delta = m_hity_1 * m_hity_xx - m_hity_x * m_hity_x;
  m_segdydz = (m_hity_1 * m_hity_xy - m_hity_x * m_hity_y) / delta;

  delta = m_hitx_1 * m_hitx_xx - m_hitx_x * m_hitx_x;
  m_segx = (m_hitx_xx * m_hitx_y - m_hitx_x * m_hitx_xy) / delta;

  delta = m_hity_1 * m_hity_xx - m_hity_x * m_hity_x;
  m_segy = (m_hity_xx * m_hity_y - m_hity_x * m_hity_xy) / delta;
}

