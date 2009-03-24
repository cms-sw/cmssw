#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "CastorGeometryData.h"

CastorGeometry::CastorGeometry() :
   theTopology( new CastorTopology ), 
   lastReqDet_(DetId::Detector(0)),
   lastReqSubdet_(0),
   m_ownsTopology ( true )
{
}

CastorGeometry::CastorGeometry( const CastorTopology* topology ) :
   theTopology(topology), 
   lastReqDet_(DetId::Detector(0)),
   lastReqSubdet_(0),
   m_ownsTopology ( false )
{
}


CastorGeometry::~CastorGeometry() 
{
   if( m_ownsTopology ) delete theTopology ;
}

const std::vector<DetId>& 
CastorGeometry::getValidDetIds( DetId::Detector det,
				int             subdet ) const 
{
   const std::vector<DetId>& baseIds ( CaloSubdetectorGeometry::getValidDetIds() ) ;
   if( det    == DetId::Detector( 0 ) &&
       subdet == 0                        )
   {
      return baseIds ;
   }
   
   if( lastReqDet_    != det    ||
       lastReqSubdet_ != subdet    ) 
   {
      lastReqDet_     = det    ;
      lastReqSubdet_  = subdet ;
      m_validIds.clear();
      m_validIds.reserve( baseIds.size() ) ;
   }

   if( m_validIds.empty() ) 
   {
      for( int i ( 0 ) ; i != baseIds.size() ; ++i ) 
      {
	 const DetId id ( baseIds[i] );
	 if( id.det()      == det    &&
	     id.subdetId() == subdet    )
	 { 
	    m_validIds.push_back( id ) ;
	 }
      }
      std::sort(m_validIds.begin(),m_validIds.end());
   }
   return m_validIds;
}

/*  NOTE only draft implementation at the moment
    what about dead volumes?
*/

DetId CastorGeometry::getClosestCell(const GlobalPoint& r) const
{
  // first find the side
  double z = r.z();
//  double x = r.x();
  double y = r.y();
  double dz = 0.;
  double zt = 0.;
  double phi = r.phi();

  int zside = 0;
  if(z >= 0)
    zside = 1;
  else
    zside =-1;

  bool isPositive = false;
  if(z>0)isPositive = true;
  z = fabs(z);
  
  // figure out if it's EM or HAD section
  // EM length = 2x51.5 mm,  HAD length = 12x101 mm
  // I assume that z0 of Castor is 14385 mm (cms.xml)
  HcalCastorDetId::Section section = HcalCastorDetId::EM;
  if(z<= theZSectionBoundaries[1])section = HcalCastorDetId::EM;
  if(z>theZSectionBoundaries[2])section = HcalCastorDetId::HAD;

  ///////////
  // figure out sector: 1-16
  // in CastorGeometryData.h theSectorBoundaries define the phi range of sectors
  //////////////


  int sector = -1;

  for( unsigned int i ( 1 ) ; i !=17 ; ++i )
  {
     if( theSectorBoundaries[i-1] <= phi &&
	 theSectorBoundaries[i  ] >  phi    ) sector = i;
  }
//  if( theSectorBoundaries[15]<= phi) sector =16;

//figure out module, just a draft for checks
  int module = -1;

// NOTE check 
 if(section ==HcalCastorDetId::EM){
  if(fabs(y) > dYEMPlate*sin(tiltangle))
      dz = (y > 0.) ?  dYEMPlate*cos(tiltangle) : -  dYEMPlate*sin(tiltangle);
    else
      dz = (y > 0.) ?  y/tan(tiltangle) : -y/tan(tiltangle);
    zt = z - dz;
    if(theZSectionBoundaries[1]<= zt <theZSectionBoundaries[2]) module = 1;
    if(zt > (theZSectionBoundaries[1]-51.5) )module = 2;
  }

  if(section == HcalCastorDetId::HAD){
    if(fabs(y) > dYHADPlate*sin(tiltangle))
      dz = (y > 0.) ?  dYHADPlate*cos(tiltangle) : -  dYHADPlate*sin(tiltangle);
    else
      dz = (y > 0.) ?  y/tan(tiltangle) : -y/tan(tiltangle);
    zt = z - dz;
    if(zt< theHadmodulesBoundaries[1]) module = 1;
    if(theHadmodulesBoundaries[1]<= zt <theHadmodulesBoundaries[2]) module = 2;
    if(theHadmodulesBoundaries[2]<= zt <theHadmodulesBoundaries[3]) module = 3;
    if(theHadmodulesBoundaries[3]<= zt <theHadmodulesBoundaries[4]) module = 4;
    if(theHadmodulesBoundaries[4]<= zt <theHadmodulesBoundaries[5]) module = 5;
    if(theHadmodulesBoundaries[5]<= zt <theHadmodulesBoundaries[6]) module = 6;
    if(theHadmodulesBoundaries[6]<= zt <theHadmodulesBoundaries[7]) module = 7;
    if(theHadmodulesBoundaries[7]<= zt <theHadmodulesBoundaries[8]) module = 8;
    if(theHadmodulesBoundaries[8]<= zt <theHadmodulesBoundaries[9]) module = 9;
    if(theHadmodulesBoundaries[9]<= zt <theHadmodulesBoundaries[10]) module = 10;
    if(theHadmodulesBoundaries[10]<= zt <theHadmodulesBoundaries[11]) module = 11;
    if(theHadmodulesBoundaries[11]<= zt ) module = 12;
  }
  
  HcalCastorDetId bestId  = HcalCastorDetId(section,isPositive, sector, module);
  return bestId;
}



unsigned int
CastorGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isCastor() ) ;

   unsigned int index ( 0 ) ;// to be implemented

   return index ;
}

unsigned int
CastorGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int)DetId::Calo ;
}

std::vector<HepPoint3D> 
CastorGeometry::localCorners( const double* pv,
			      unsigned int  i,
			      HepPoint3D&   ref )
{
   return ( calogeom::IdealCastorTrapezoid::localCorners( pv, ref ) ) ;
}

CaloCellGeometry* 
CastorGeometry::newCell( const GlobalPoint& f1 ,
			 const GlobalPoint& f2 ,
			 const GlobalPoint& f3 ,
			 CaloCellGeometry::CornersMgr* mgr,
			 const double*      parm ,
			 const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;
   
   assert( cgid.isCastor() ) ;

   return ( new calogeom::IdealCastorTrapezoid( f1, mgr, parm ) ) ;
}
