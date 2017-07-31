#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template<>
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::produceAligned( const typename HGCalGeometry::AlignedRecord& iRecord ) 
{
  TrVec  tvec;
  DimVec dvec;
  IVec   ivec;
  IVec   dins;

  std::string name;

  name = "HGCalEESensitive";
  std::cout << "Reading HGCalGeometry " << name.c_str() << "\n";
  if( CaloGeometryDBReader::writeFlag() )
  {
    edm::ESHandle<HGCalGeometry> geom;
    iRecord.getRecord<IdealGeometryRecord>().get( name, geom );
     
    geom->getSummary( tvec, ivec, dvec, dins ) ;

    CaloGeometryDBReader::writeIndexed( tvec, dvec, ivec, dins, HGCalGeometry::dbString() ) ;
  }
  else
  {
    edm::ESHandle<PCaloGeometry> pG ;
    iRecord.getRecord<typename HGCalGeometry::PGeometryRecord >().get( pG ) ; 
    
    tvec = pG->getTranslation() ;
    dvec = pG->getDimension() ;
    ivec = pG->getIndexes() ;
    dins = pG->getDenseIndices();
  }	 
  //*********************************************************************************************
  edm::ESHandle<HGCalTopology> topology;
  iRecord.getRecord<IdealGeometryRecord>().get( name, topology );

  assert( dvec.size() <= topology->totalGeomModules() * HGCalGeometry::k_NumberOfParametersPerShape );
  HGCalGeometry* hcg = new HGCalGeometry( *topology );
  PtrType ptr ( hcg );

  ptr->allocateCorners( topology->ncells());
  ptr->allocatePar( HGCalGeometry::k_NumberOfShapes,
		    HGCalGeometry::k_NumberOfParametersPerShape );

  const unsigned int nTrParm( ptr->numberOfTransformParms());
  const unsigned int nPerShape( HGCalGeometry::k_NumberOfParametersPerShape );
 
  for( auto it : dins )
  {
    DetId id = topology->encode( topology->geomDenseId2decId( it ));
    // get layer
    int layer = ivec[ it ];

    // get transformation
    const unsigned int jj ( it * nTrParm );
    Tr3D tr;
    const ROOT::Math::Translation3D tl( tvec[jj], tvec[jj+1], tvec[jj+2]);
    const ROOT::Math::EulerAngles ea( 6 == nTrParm ?
				      ROOT::Math::EulerAngles( tvec[jj+3], tvec[jj+4], tvec[jj+5] ) :
				      ROOT::Math::EulerAngles());
    const ROOT::Math::Transform3D rt( ea, tl );
    double xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz;
    rt.GetComponents(xx,xy,xz,dx,yx,yy,yz,dy,zx,zy,zz,dz) ;
    tr = Tr3D( CLHEP::HepRep3x3( xx, xy, xz,
				 yx, yy, yz,
				 zx, zy, zz ), 
	       CLHEP::Hep3Vector( dx, dy, dz));
    
    // get parameters
    DimVec dims;
    dims.reserve( nPerShape );
    
    DimVec::const_iterator dsrc( dvec.begin() + layer * nPerShape );
    for( unsigned int j ( 0 ) ; j != nPerShape ; ++j )
    {
      dims.emplace_back( *dsrc ) ;
      ++dsrc ;
    }
    
    std::vector<GlobalPoint> corners( 8 );
    
    FlatTrd::createCorners( dims, tr, corners );
    
    const CCGFloat* myParm( CaloCellGeometry::getParmPtr( dims, 
							  ptr->parMgr(), 
							  ptr->parVecVec()));
    GlobalPoint front ( 0.25*( corners[0].x() + 
			       corners[1].x() + 
			       corners[2].x() + 
			       corners[3].x()),
			0.25*( corners[0].y() + 
			       corners[1].y() + 
			       corners[2].y() + 
			       corners[3].y()),
			0.25*( corners[0].z() + 
			       corners[1].z() + 
			       corners[2].z() + 
			       corners[3].z()));
    
    GlobalPoint back  ( 0.25*( corners[4].x() + 
			       corners[5].x() + 
			       corners[6].x() + 
			       corners[7].x()),
			0.25*( corners[4].y() + 
			       corners[5].y() + 
			       corners[6].y() + 
			       corners[7].y()),
			0.25*( corners[4].z() + 
			       corners[5].z() + 
			       corners[6].z() + 
			       corners[7].z()));
    
    if (front.mag2() > back.mag2()) { // front should always point to the center, so swap front and back
      std::swap (front, back);
      std::swap_ranges (corners.begin(), corners.begin()+4, corners.begin()+4); 
    }
    
    ptr->newCell( front, back, corners[0], myParm, id );
  }
  
  ptr->initializeParms(); // initializations; must happen after cells filled
  
  return ptr ; 
}

template class CaloGeometryDBEP< HGCalGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< HGCalGeometry , CaloGeometryDBReader> 
HGCalGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HGCalGeometryFromDBEP);

