/* 
 * $Id: MuonTrackCSCChamberResidual.cc,v 1.4 2013/01/02 15:07:22 eulisse Exp $
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonTrackCSCChamberResidual.h"
//#include "Geometry/CSCGeometry/interface/CSCGeometry.h"


MuonTrackCSCChamberResidual::MuonTrackCSCChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry, AlignableNavigator *navigator,
                                                         DetId chamberId, AlignableDetOrUnitPtr chamberAlignable)
  : MuonChamberResidual(globalGeometry, navigator, chamberId, chamberAlignable)
{
  m_type = MuonChamberResidual::kCSC;
  align::GlobalVector zDirection(0., 0., 1.);
  m_sign = m_globalGeometry->idToDet(m_chamberId)->toLocal(zDirection).z() > 0. ? 1. : -1.;
}


void MuonTrackCSCChamberResidual::setSegmentResidual(const reco::MuonChamberMatch *trk, const reco::MuonSegmentMatch *seg)
{
  CSCDetId id(trk->id.rawId());
  
  CSCSegmentRef segmentCSC = seg->cscSegmentRef;
  if (segmentCSC.get() != 0)
  {
    const CSCSegment* segment = segmentCSC.get();
    m_numHits = segment->nRecHits();
    m_ndof = segment->degreesOfFreedom();
    m_chi2 = segment->chi2();
    //std::cout<<"csc seg position = "<<segment->localPosition()<<"  numhits="<<m_numHits<<"  id: "<<id<<std::endl;
  }

  align::LocalPoint l_seg(seg->x, seg->y, 0.);
  align::LocalPoint l_trk(trk->x, trk->y, 0.);
  //align::GlobalPoint g_seg = m_globalGeometry->idToDet(chamber)->toGlobal(l_seg);
  //align::GlobalPoint g_trk = m_globalGeometry->idToDet(chamber)->toGlobal(l_trk);

  /*
  double dphi = g_trk.phi() - g_seg.phi();
  while (dphi >  M_PI) dphi -= 2.*M_PI;
  while (dphi < -M_PI) dphi += 2.*M_PI;
  m_residual = - m_sign * g_trk.perp() * dphi; // coming from global, need to adjust the sign
  std::cout<<"cscres="<<m_residual<<"  dx="<<trk->x-seg->x<<"  diff="<<trk->x-seg->x - m_residual<<std::endl;
  */
  m_residual = trk->x-seg->x;
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

  //double yresidual_error = sqrt( pow(trk->yErr, 2) + pow(seg->yErr, 2) );
  //double yresslope_error = sqrt( pow(trk->dYdZErr, 2) + pow(seg->dYdZErr, 2) );
  //std::cout<<"csc res "<<m_residual<<"+-"<<m_residual_error<<"  "<<m_resslope<<"+-"<<m_resslope_error<<"  "<<trk->y-seg->y<<"+-"<<yresidual_error<<"  "<<trk->dYdZ - seg->dYdZ<<"+-"<<yresslope_error<<std::endl;
  //std::cout<<"csc trk "<<m_trackx<<" "<<m_tracky<<" "<<m_trackdxdz<<" "<<m_trackdydz<<std::endl;
  //std::cout<<"csc seg "<<m_segx<<" "<<m_segy<<" "<<m_segdxdz<<" "<<m_segdydz<<std::endl;
}
