/*
 * $Id: MuonTrackDT13ChamberResidual.cc,v 1.1 2011/10/12 23:32:08 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT13ChamberResidual.h"


MuonTrackDT13ChamberResidual::MuonTrackDT13ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                                                           DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
  : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
{
  m_type = MuonChamberResidual::kDT13;
  double rphiAngle = atan2(m_globalGeometry->idToDet(m_chamberId)->position().y(), m_globalGeometry->idToDet(m_chamberId)->position().x()) + M_PI/2.;
  align::GlobalVector rphiDirection(cos(rphiAngle), sin(rphiAngle), 0.);
  m_sign = m_globalGeometry->idToDet(m_chamberId)->toLocal(rphiDirection).x() > 0. ? 1. : -1.;
}


void MuonTrackDT13ChamberResidual::setSegmentResidual(const reco::MuonChamberMatch *trk, const reco::MuonSegmentMatch *seg)
{
  DTRecSegment4DRef segmentDT = seg->dtSegmentRef;
  if (segmentDT.get() != 0)
  {
    const DTRecSegment4D* segment = segmentDT.get();
    assert(segment->hasPhi());
    const DTChamberRecSegment2D* phiSeg = segment->phiSegment();
    m_numHits = phiSeg->recHits().size();
    m_ndof = phiSeg->degreesOfFreedom();
    m_chi2 = phiSeg->chi2();
    //std::cout<<"phi seg position = "<<phiSeg->localPosition()<<"  numhits="<<m_numHits<<std::endl;
  }

  m_residual = trk->x - seg->x;
  m_residual_error = sqrt( pow(trk->xErr, 2) + pow(seg->xErr, 2) );
  m_resslope = trk->dXdZ - seg->dXdZ;
  m_resslope_error = sqrt( pow(trk->dXdZErr, 2) + pow(seg->dXdZErr, 2) );
  
  m_trackx = trk->x;
  m_tracky = trk->y;
  m_trackdxdz = trk->dXdZ;
  m_trackdydz = trk->dYdZ;

  m_segx = seg->x;
  m_segy = seg->y;
  m_segdxdz = seg->dXdZ;
  m_segdydz = seg->dYdZ;
  
  //std::cout<<"d13 res "<<m_residual<<"+-"<<m_residual_error<<"  "<<m_resslope<<"+-"<<m_resslope_error<<std::endl;
  //std::cout<<"d13 trk "<<m_trackx<<" "<<m_tracky<<" "<<m_trackdxdz<<" "<<m_trackdydz<<std::endl;
  //std::cout<<"d13 seg "<<m_segx<<" "<<m_segy<<" "<<m_segdxdz<<" "<<m_segdydz<<std::endl;
}
