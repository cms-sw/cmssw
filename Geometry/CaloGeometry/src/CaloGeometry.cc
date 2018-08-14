#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

const std::vector<DetId> CaloGeometry::k_emptyVec ( 0 ) ;

CaloGeometry::CaloGeometry() :
   m_geos ( kLength, nullptr )
{
}

unsigned int 
CaloGeometry::makeIndex( DetId::Detector det    , 
			 int             subdet ,
			 bool&           ok       ) const 
{
   const unsigned int idet ( det ) ;

   ok = ( kMinDet <= idet   &&
	  kMaxDet >= idet   &&
	  0       <= subdet &&
	  kMaxSub >= subdet    ) ;
   if (!ok)
     edm::LogWarning("CaloGeometry") << "Det:Subdet " << idet << ":" << subdet 
				     << " min|max Det " << kMinDet << ":" 
				     << kMaxDet << " min|max subdet 0:" <<kMaxSub;
   
   return ( ( det - kMinDet ) * kNSubDets + subdet ) ;
}

void 
CaloGeometry::setSubdetGeometry( DetId::Detector                det    , 
				 int                            subdet , 
				 const CaloSubdetectorGeometry* geom     )  {
   bool ok ;
   const unsigned int index = makeIndex( det, subdet, ok ) ;
   if( ok ) m_geos[index] = geom ;

   edm::LogVerbatim("CaloGeometry") << "Detector=" << (int)det << ", subset="
				    << subdet << ", index=" << index
				    << ", size=" << m_geos.size();

   assert( ok ) ;
}

const CaloSubdetectorGeometry* 
CaloGeometry::getSubdetectorGeometry( const DetId& id ) const 
{
   bool ok ;

   const unsigned int index ( makeIndex( id.det(),
					 id.subdetId(),
					 ok             ) ) ;
   return ( ok ? m_geos[ index ] : nullptr ) ;
}

const CaloSubdetectorGeometry* 
CaloGeometry::getSubdetectorGeometry( DetId::Detector det    , 
				      int             subdet  ) const 
{
   bool ok ;

   const unsigned int index ( makeIndex( det,
					 subdet,
					 ok             ) ) ;
   return ( ok ? m_geos[ index ] : nullptr ) ;
}

static const GlobalPoint notFound(0,0,0);

GlobalPoint
CaloGeometry::getPosition( const DetId& id ) const {
  const CaloSubdetectorGeometry* geom = getSubdetectorGeometry( id ) ;
  if (geom) {
    GlobalPoint pos = geom->getGeometry(id)->getPosition();
    return pos;
  } else {
    return notFound;
  }
}

std::shared_ptr<const CaloCellGeometry>
CaloGeometry::getGeometry( const DetId& id ) const {
  const CaloSubdetectorGeometry* geom = getSubdetectorGeometry(id);
  if (geom) {
    auto cell = geom->getGeometry(id);
    return cell;
  } else {
    return std::shared_ptr<const CaloCellGeometry>();
  }
}

bool 
CaloGeometry::present( const DetId& id ) const
{
  const CaloSubdetectorGeometry* geom = getSubdetectorGeometry(id) ;
  return ( nullptr == geom ? false : geom->present( id ) ) ;
}

std::vector<DetId> CaloGeometry::getValidDetIds() const
{
   std::vector<DetId> returnValue ;
   returnValue.reserve( kLength ) ;

   bool doneHcal ( false ) ;
   for( unsigned int i ( 0 ) ; i != m_geos.size() ; ++i ) 
   {     
      if( nullptr != m_geos[i] )
      {
	 const std::vector< DetId >& aVec = m_geos[i]->getValidDetIds();	 
	 if( aVec.empty() ) {
	   edm::LogWarning("CaloGeometry") << "Valid det id list at index " 
					   << i << " is empty!";
	 }
	 const bool isHcal ( !aVec.empty() && DetId::Hcal == aVec.front().det() ) ;
	 if( !doneHcal ||
	     !isHcal      )
	 {
	    returnValue.insert( returnValue.end(), aVec.begin(), aVec.end() ) ;
	    if( !doneHcal &&
		isHcal        ) doneHcal = true ;
	 }
      }
   }
   return returnValue ;
}

const std::vector<DetId>&
CaloGeometry::getValidDetIds( DetId::Detector det    , 
			      int             subdet  ) const 
{
   bool ok ;

   const unsigned int index ( makeIndex( det,
					 subdet,
					 ok             ) ) ;

   return ( ok && ( nullptr != m_geos[ index ] ) ?
	    m_geos[ index ]->getValidDetIds( det, subdet ) :
	    k_emptyVec ) ;
}
  

