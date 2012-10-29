#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

//#define DebugLog

HcalDDDGeometry::HcalDDDGeometry(const HcalTopology& topo) : topo_(topo),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   etaMax_(0),
   firstHFQuadRing_(40) ,
							     m_hbCellVec ( topo.getHBSize() ) ,
							     m_heCellVec ( topo.getHESize() ) ,
							     m_hoCellVec ( topo.getHOSize() ) ,
							     m_hfCellVec ( topo.getHFSize() ) 
{
  twopi = M_PI + M_PI;
  deg   = M_PI/180.;
}


HcalDDDGeometry::~HcalDDDGeometry()
{
}


std::vector<DetId> const & HcalDDDGeometry::getValidDetIds(DetId::Detector det,
							   int subdet) const {

  const std::vector<DetId>& baseIds(CaloSubdetectorGeometry::getValidDetIds());
  if (det    == DetId::Detector( 0 ) && subdet == 0) {
    return baseIds ;
  }
   
  if (lastReqDet_  != det || lastReqSubdet_ != subdet ) {
    lastReqDet_     = det    ;
    lastReqSubdet_  = subdet ;
    m_validIds.clear();
    m_validIds.reserve( baseIds.size() ) ;
  }

  if (m_validIds.empty() ) {
    for (unsigned int i = 0 ; i != baseIds.size() ; ++i ) {
      const DetId id ( baseIds[i] );
      if (id.det()      == det   && id.subdetId() == subdet ) { 
	m_validIds.push_back( id ) ;
      }
    }
    std::sort(m_validIds.begin(),m_validIds.end());
  }
  
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalDDDGeometry::getValidDetIds: "
		       << m_validIds.size() << " valid IDs found for detector "
		       << det << " Sub-detector " << subdet;
#endif
  return m_validIds;
}


DetId HcalDDDGeometry::getClosestCell(const GlobalPoint& r) const {

  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  double abseta = fabs(r.eta());
  double phi    = r.phi();
  if (phi < 0) phi += twopi;
  double radius = r.mag();
  double z      = fabs(r.z());
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell for eta "
		       << r.eta() << " phi " << phi/deg << " z " << r.z()
		       << " radius " << radius;
#endif

  HcalDetId bestId;
  if (abseta <= etaMax_) {
    for (unsigned int i=0; i<hcalCells_.size(); i++) {
      if (abseta >=hcalCells_[i].etaMin() && abseta <=hcalCells_[i].etaMax()) {
	HcalSubdetector bc = hcalCells_[i].detType();
	int etaring = hcalCells_[i].etaBin();
	int phibin  = 0;
	if (hcalCells_[i].unitPhi() == 4) {
	  // rings 40 and 41 are offset wrt the other phi numbering
	  //  1        1         1         2
	  //  ------------------------------
	  //  72       36        36        1
	  phibin = static_cast<int>(((phi/deg)+hcalCells_[i].phiOffset()+
				     0.5*hcalCells_[i].phiBinWidth())/
				    hcalCells_[i].phiBinWidth());
	  if (phibin == 0) phibin = hcalCells_[i].nPhiBins();
	  phibin = phibin*4 - 1; 
	} else {
	  phibin = static_cast<int>(((phi/deg)+hcalCells_[i].phiOffset())/
				    hcalCells_[i].phiBinWidth()) + 1;
	  // convert to the convention of numbering 1,3,5, in 36 phi bins
	  phibin = (phibin-1)*(hcalCells_[i].unitPhi()) + 1;
	}

	int dbin   = 1;
	int etabin = (r.z() > 0) ? etaring : -etaring;
	if (bc == HcalForward) {
	  bestId   = HcalDetId(bc, etabin, phibin, dbin);
	  break;
	} else {
	  double rz = z;
	  if (hcalCells_[i].depthType()) rz = radius;
	  if (rz < hcalCells_[i].depthMax()) {
	    dbin   = hcalCells_[i].depthSegment();
	    bestId = HcalDetId(bc, etabin, phibin, dbin);
	    break;
	  }
	}
      }
    }
  }
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell " << bestId;
#endif
  return bestId;
}


int HcalDDDGeometry::insertCell(std::vector<HcalCellType> const & cells){

  hcalCells_.insert(hcalCells_.end(), cells.begin(), cells.end());
  int num = static_cast<int>(hcalCells_.size());
  for (unsigned int i=0; i<cells.size(); i++) {
    if (cells[i].etaMax() > etaMax_ ) etaMax_ = cells[i].etaMax();
  }
#ifdef DebugLog
  LogDebug("HCalGeom") << "HcalDDDGeometry::insertCell " << cells.size()
		       << " cells inserted == Total " << num
		       << " EtaMax = " << etaMax_;
#endif
  return num;
}

void
HcalDDDGeometry::newCell( const GlobalPoint& f1 ,
			  const GlobalPoint& f2 ,
			  const GlobalPoint& f3 ,
			  const CCGFloat*    parm ,
			  const DetId&       detId   ) 
{

   assert( detId.det()==DetId::Hcal );
  
   const unsigned int din(topo_.detId2denseId(detId));

   HcalDetId hId(detId);

   if( hId.subdet()==HcalBarrel )
   {
      m_hbCellVec[ din ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
   }
   else
   {
     if( hId.subdet()==HcalEndcap )
      {
	 const unsigned int index ( din - m_hbCellVec.size() ) ;
	 m_heCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
      }
      else
      {
	if( hId.subdet()==HcalOuter )
	 {
	    const unsigned int index ( din 
				       - m_hbCellVec.size() 
				       - m_heCellVec.size() ) ;
	    m_hoCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
	 }
	 else
	   { // assuming HcalForward here!
	    const unsigned int index ( din 
				       - m_hbCellVec.size() 
				       - m_heCellVec.size() 
				       - m_hoCellVec.size() ) ;
	    m_hfCellVec[ index ] = IdealZPrism( f1, cornersMgr(), parm ) ;
	 }
      }
   }
   m_validIds.push_back( detId ) ;
}

const CaloCellGeometry* 
HcalDDDGeometry::cellGeomPtr( uint32_t din ) const
{
   const CaloCellGeometry* cell ( 0 ) ;
   if( m_hbCellVec.size() > din )
   {
      cell = &m_hbCellVec[ din ] ;
   }
   else
   {
      if( m_hbCellVec.size() +
	  m_heCellVec.size() > din )
      {
	 const unsigned int index ( din - m_hbCellVec.size() ) ;
	 cell = &m_heCellVec[ index ] ;
      }
      else
      {
	 if( m_hbCellVec.size() +
	     m_heCellVec.size() +
	     m_hoCellVec.size() > din )
	 {
	    const unsigned int index ( din 
				       - m_hbCellVec.size() 
				       - m_heCellVec.size() ) ;
	    cell = &m_hoCellVec[ index ] ;
	 }
	 else
	 {
	    if( m_hbCellVec.size() +
		m_heCellVec.size() +
		m_hoCellVec.size() +
		m_hfCellVec.size() > din )
	    {
	       const unsigned int index ( din 
					  - m_hbCellVec.size() 
					  - m_heCellVec.size() 
					  - m_hoCellVec.size() ) ;
	       cell = &m_hfCellVec[ index ] ;
	    }
	 }
      }
   }
   return ( 0 == cell || 0 == cell->param() ? 0 : cell ) ;
}
