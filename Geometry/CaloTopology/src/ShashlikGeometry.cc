#include "Geometry/CaloTopology/interface/ShashlikGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ShashlikGeometry::ShashlikGeometry( const ShashlikTopology& topology )
  : m_topology( topology ),
    m_ekCellVec( EKCellVec( 1400 )) // FIXME: Shashlik topology should know how many DetIds there will be.
{}

ShashlikGeometry::~ShashlikGeometry( void ) 
{}

void
ShashlikGeometry::newCell( const GlobalPoint& f1,
			   const GlobalPoint& f2,
			   const GlobalPoint& f3,
			   const CCGFloat*    parm,
			   const DetId&       detId )
{
  assert( detId.det() == DetId::Forward ); //FIXME: Is it?

  const EKDetId ekid( detId );
  unsigned int din = m_topology.detId2denseId( detId );

  edm::LogInfo("ShashlikGeometry") << " newCell subdet "
				   << detId.subdetId() << ", raw ID " 
				   << detId.rawId() << ", hid " << ekid << ", din " 
				   << din << ", index ";

  m_ekCellVec[ din ] = IdealObliquePrism( f1, cornersMgr(), parm );
  addValidID( detId );
  m_dins.push_back( din );
}

const CaloCellGeometry* 
ShashlikGeometry::cellGeomPtr( unsigned int din ) const
{
  const CaloCellGeometry* cell( 0 );

  assert( m_ekCellVec.size() > din );
  cell = &m_ekCellVec[ din ];

  return(( 0 == cell || 0 == cell->param()) ? 0 : cell );
}
