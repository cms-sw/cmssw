/** \class MuonChamberResidual
 *  $Id: MuonChamberResidual.cc,v 1.1 2011/10/12 23:32:08 khotilov Exp $
 *  \author V. Khotilovich - Texas A&M University <khotilov@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"


MuonChamberResidual::MuonChamberResidual(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                                         AlignableNavigator *navigator, 
                                         DetId chamberId,
                                         AlignableDetOrUnitPtr chamberAlignable):
    m_globalGeometry(globalGeometry)
  , m_navigator(navigator)
  , m_chamberId(chamberId)
  , m_chamberAlignable(chamberAlignable)
  , m_numHits(0)
  , m_type(-1)
  , m_sign(0.)
  , m_chi2(-999.)
  , m_ndof(-1)
  , m_residual(-999.)
  , m_residual_error(-999.)
  , m_resslope(-999.)
  , m_resslope_error(-999.)
  , m_trackdxdz(-999.)
  , m_trackdydz(-999.)
  , m_trackx(-999.)
  , m_tracky(-999.)
  , m_segdxdz(-999.)
  , m_segdydz(-999.)
  , m_segx(-999.)
  , m_segy(-999.)
{}


align::GlobalPoint MuonChamberResidual::global_trackpos()
{
  return chamberAlignable()->surface().toGlobal(align::LocalPoint(trackx(), tracky(), 0.));
}


align::GlobalPoint MuonChamberResidual::global_stubpos()
{
  return chamberAlignable()->surface().toGlobal(align::LocalPoint(segx(), segy(), 0.));
}


double MuonChamberResidual::global_residual() const 
{
  return residual() * signConvention();
}


double MuonChamberResidual::global_resslope() const
{
  return resslope() * signConvention();
}


double MuonChamberResidual::global_hitresid(int i) const
{
  return hitresid(i) * signConvention();
}


double MuonChamberResidual::hitresid(int i) const
{
  assert(0 <= i  &&  i < int(m_localIDs.size()));
  return m_localResids[i];
}


int MuonChamberResidual::hitlayer(int i) const 
{  // only difference between DTs and CSCs is the DetId subclass
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
}


double MuonChamberResidual::hitposition(int i) const
{
  assert(0 <= i  &&  i < int(m_localIDs.size()));
  if (m_chamberId.subdetId() == MuonSubdetId::DT) {
    align::GlobalPoint pos = m_globalGeometry->idToDet(m_localIDs[i])->position();
    return sqrt(pow(pos.x(), 2) + pow(pos.y(), 2));                   // R for DTs
  }
  else if (m_chamberId.subdetId() == MuonSubdetId::CSC) {
    return m_globalGeometry->idToDet(m_localIDs[i])->position().z();  // Z for CSCs
  }
  else assert(false);
}
