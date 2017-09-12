#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalDDDGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <mutex>

static std::mutex s_fillLock;

HcalDDDGeometry::HcalDDDGeometry(const HcalTopology& topo)
  : topo_(topo),
    etaMax_(0),
    m_hbCellVec ( topo.getHBSize() ) ,
    m_heCellVec ( topo.getHESize() ) ,
    m_hoCellVec ( topo.getHOSize() ) ,
    m_hfCellVec ( topo.getHFSize() ) ,
    m_filledDetIds(false)
{
}

HcalDDDGeometry::~HcalDDDGeometry()
{
}

void
HcalDDDGeometry::fillDetIds() const
{
   std::lock_guard<std::mutex> guard(s_fillLock);
   if (m_filledDetIds) {
     //another thread already did the work
     return;
   }
   const std::vector<DetId>& baseIds ( CaloSubdetectorGeometry::getValidDetIds() ) ;
   for( unsigned int i ( 0 ) ; i != baseIds.size() ; ++i ) 
   {
      const DetId id ( baseIds[i] );
      if( id.subdetId() == HcalBarrel )
      { 
	 m_hbIds.emplace_back( id ) ;
      }
      else
      {
	 if( id.subdetId() == HcalEndcap )
	 { 
	    m_heIds.emplace_back( id ) ;
	 }
	 else
	 {
	    if( id.subdetId() == HcalOuter )
	    { 
	       m_hoIds.emplace_back( id ) ;
	    }
	    else
	    {
	       if( id.subdetId() == HcalForward )
	       { 
		  m_hfIds.emplace_back( id ) ;
	       }
	    }
	 }
      }
   }
   std::sort( m_hbIds.begin(), m_hbIds.end() ) ;
   std::sort( m_heIds.begin(), m_heIds.end() ) ;
   std::sort( m_hoIds.begin(), m_hoIds.end() ) ;
   std::sort( m_hfIds.begin(), m_hfIds.end() ) ;
       
   m_emptyIds.resize( 0 ) ;
   m_filledDetIds = true;
}

std::vector<DetId> const &
HcalDDDGeometry::getValidDetIds(DetId::Detector det,
				int subdet) const
{
  if( 0 != subdet &&
      not m_filledDetIds ) fillDetIds() ;
  return ( 0 == subdet ? CaloSubdetectorGeometry::getValidDetIds() :
	   ( HcalBarrel == subdet ? m_hbIds :
	     ( HcalEndcap == subdet ? m_heIds :
	       ( HcalOuter == subdet ? m_hoIds :
		 ( HcalForward == subdet ? m_hfIds : m_emptyIds ) ) ) ) ) ;
}

DetId
HcalDDDGeometry::getClosestCell(const GlobalPoint& r) const
{
  constexpr double twopi = M_PI+M_PI;
  constexpr double deg   = M_PI/180.;

  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  double abseta = fabs(r.eta());
  double phi    = r.phi();
  if (phi < 0) phi += twopi;
  double radius = r.mag();
  double z      = fabs(r.z());

  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell for eta "
		       << r.eta() << " phi " << phi/deg << " z " << r.z()
		       << " radius " << radius;
  HcalDetId bestId;
  if (abseta <= etaMax_) {
    for (const auto & hcalCell : hcalCells_) {
      if (abseta >=hcalCell.etaMin() && abseta <=hcalCell.etaMax()) {
	HcalSubdetector bc = hcalCell.detType();
	int etaring = hcalCell.etaBin();
	int phibin  = 0;
	if (hcalCell.unitPhi() == 4) {
	  // rings 40 and 41 are offset wrt the other phi numbering
	  //  1        1         1         2
	  //  ------------------------------
	  //  72       36        36        1
	  phibin = static_cast<int>((phi+hcalCell.phiOffset()+
				     0.5*hcalCell.phiBinWidth())/
				    hcalCell.phiBinWidth());
	  if (phibin == 0) phibin = hcalCell.nPhiBins();
	  phibin = phibin*4 - 1; 
	} else {
	  phibin = static_cast<int>((phi+hcalCell.phiOffset())/
				    hcalCell.phiBinWidth()) + 1;
	  // convert to the convention of numbering 1,3,5, in 36 phi bins
	  phibin = (phibin-1)*(hcalCell.unitPhi()) + 1;
	}

	int dbin   = 1;
	int etabin = (r.z() > 0) ? etaring : -etaring;
	if (bc == HcalForward) {
	  bestId   = HcalDetId(bc, etabin, phibin, dbin);
	  break;
	} else {
	  double rz = z;
	  if (hcalCell.depthType()) rz = radius;
	  if (rz < hcalCell.depthMax()) {
	    dbin   = hcalCell.depthSegment();
	    bestId = HcalDetId(bc, etabin, phibin, dbin);
	    break;
	  }
	}
      }
    }
  }

  LogDebug("HCalGeom") << "HcalDDDGeometry::getClosestCell " << bestId;

  return bestId;
}

int
HcalDDDGeometry::insertCell(std::vector<HcalCellType> const & cells){

  hcalCells_.insert(hcalCells_.end(), cells.begin(), cells.end());
  int num = static_cast<int>(hcalCells_.size());
  for (const auto & cell : cells) {
    if (cell.etaMax() > etaMax_ ) etaMax_ = cell.etaMax();
  }

  LogDebug("HCalGeom") << "HcalDDDGeometry::insertCell " << cells.size()
		       << " cells inserted == Total " << num
		       << " EtaMax = " << etaMax_;
  return num;
}

void
HcalDDDGeometry::newCellImpl( const GlobalPoint& f1 ,
			  const GlobalPoint& f2 ,
			  const GlobalPoint& f3 ,
			  const CCGFloat*    parm ,
			  const DetId&       detId   ) 
{

  assert( detId.det()==DetId::Hcal );
  
  const unsigned int din(topo_.detId2denseId(detId));

  HcalDetId hId(detId);

  if( hId.subdet()==HcalBarrel ) {
    m_hbCellVec[ din ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
  } else {
    if( hId.subdet()==HcalEndcap ) {
      const unsigned int index ( din - m_hbCellVec.size() ) ;
      m_heCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
    } else  {
      if( hId.subdet()==HcalOuter )  {
	const unsigned int index ( din 
				   - m_hbCellVec.size() 
				   - m_heCellVec.size() ) ;
	m_hoCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
      } else { // assuming HcalForward here!
	const unsigned int index ( din 
				   - m_hbCellVec.size() 
				   - m_heCellVec.size() 
				   - m_hoCellVec.size() ) ;
	m_hfCellVec[ index ] = IdealZPrism( f1, cornersMgr(), parm, hId.depth()==1 ? IdealZPrism::EM : IdealZPrism::HADR ) ;
      }
    }
  }
}

void
HcalDDDGeometry::newCell( const GlobalPoint& f1 ,
              const GlobalPoint& f2 ,
              const GlobalPoint& f3 ,
              const CCGFloat*    parm ,
              const DetId&       detId   )
{
  newCellImpl(f1,f2,f3,parm,detId);
  addValidID( detId );
}

void
HcalDDDGeometry::newCellFast( const GlobalPoint& f1 ,
              const GlobalPoint& f2 ,
              const GlobalPoint& f3 ,
              const CCGFloat*    parm ,
              const DetId&       detId   )
{
  newCellImpl(f1,f2,f3,parm,detId);
  m_validIds.emplace_back(detId);
}

const CaloCellGeometry* 
HcalDDDGeometry::cellGeomPtr( uint32_t din ) const
{
  const CaloCellGeometry* cell ( nullptr ) ;
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
  return ( nullptr == cell || nullptr == cell->param() ? nullptr : cell ) ;
}

void HcalDDDGeometry::increaseReserve(unsigned int extra) {
  m_validIds.reserve(m_validIds.size()+extra);
}

void HcalDDDGeometry::sortValidIds() {
  std::sort(m_validIds.begin(),m_validIds.end());
}
