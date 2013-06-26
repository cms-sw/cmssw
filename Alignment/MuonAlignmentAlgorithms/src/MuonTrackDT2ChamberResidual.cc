/* 
 * $Id: MuonTrackDT2ChamberResidual.cc,v 1.1 2011/10/12 23:32:08 khotilov Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackDT2ChamberResidual.h"

MuonTrackDT2ChamberResidual::MuonTrackDT2ChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                                                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
  : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
{
  m_type = MuonChamberResidual::kDT2; 
  align::GlobalVector zDirection(0., 0., 1.);
  m_sign = m_globalGeometry->idToDet(m_chamberId)->toLocal(zDirection).y() > 0. ? 1. : -1.; 
}


void MuonTrackDT2ChamberResidual::setSegmentResidual(const reco::MuonChamberMatch *trk, const reco::MuonSegmentMatch *seg)
{
  DTRecSegment4DRef segmentDT = seg->dtSegmentRef;
  if (segmentDT.get() != 0)
  {
    const DTRecSegment4D* segment = segmentDT.get();
    assert(segment->hasZed());
    const DTSLRecSegment2D* zSeg = (*segment).zSegment();
    m_numHits = zSeg->recHits().size();
    m_ndof = zSeg->degreesOfFreedom();
    m_chi2 = zSeg->chi2();
    //std::cout<<"z seg position = "<<zSeg->localPosition()<<"  numhits="<<m_numHits<<std::endl;
  }
  
  m_residual = trk->y - seg->y;
  m_residual_error = sqrt( pow(trk->yErr, 2) + pow(seg->yErr, 2) );
  m_resslope = trk->dYdZ - seg->dYdZ;
  m_resslope_error = sqrt( pow(trk->dYdZErr, 2) + pow(seg->dYdZErr, 2) );
  
  m_trackx = trk->x;
  m_tracky = trk->y;
  m_trackdxdz = trk->dXdZ;
  m_trackdydz = trk->dYdZ;

  m_segx = seg->x;
  m_segy = seg->y;
  m_segdxdz = seg->dXdZ;
  m_segdydz = seg->dYdZ;

  //std::cout<<"d2 res "<<m_residual<<"+-"<<m_residual_error<<"  "<<m_resslope<<"+-"<<m_resslope_error<<std::endl;
  //std::cout<<"d2 trk "<<m_trackx<<" "<<m_tracky<<" "<<m_trackdxdz<<" "<<m_trackdydz<<std::endl;
  //std::cout<<"d2 seg "<<m_segx<<" "<<m_segy<<" "<<m_segdxdz<<" "<<m_segdydz<<std::endl;
}
