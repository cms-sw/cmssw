#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template<>
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::produceAligned( const typename HGCalGeometry::AlignedRecord& iRecord ) 
{
    TrVec  tvec ;
    DimVec dvec ;
    IVec   ivec ;
    IVec   dins ;

    std::string name;

    name = "HGCalEESensitive";

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

 
    edm::ESHandle<HGCalTopology> hgcalTopology;
    iRecord.getRecord<IdealGeometryRecord>().get( name, hgcalTopology );

    // We know that the numer of shapes chanes with changing depth
    // so, this check is temporary disabled. We need to implement
    // a way either to store or calculate the number of shapes or be able
    // to deal with only max numer of shapes.
    // assert( dvec.size() == hcalTopology->getNumberOfShapes() * HGCalGeometry::k_NumberOfParametersPerShape ) ;
    assert( dvec.size() <= hgcalTopology->totalGeomModules() * HGCalGeometry::k_NumberOfParametersPerShape ) ;
    HGCalGeometry* hcg = new HGCalGeometry( *hgcalTopology );
    
    PtrType ptr ( hcg );
 
    const unsigned int nTrParm ( tvec.size()/hgcalTopology->ncells() ) ;
   
    ptr->fillDefaultNamedParameters() ;

    ptr->allocateCorners( hgcalTopology->ncells() );

    ptr->allocatePar(    dvec.size() ,
			 HGCalGeometry::k_NumberOfParametersPerShape ) ;

    for( unsigned int i ( 0 ) ; i < dins.size() ; ++i )
    {
	const unsigned int nPerShape ( HGCalGeometry::k_NumberOfParametersPerShape ) ;
	DimVec dims ;
	dims.reserve( nPerShape ) ;

	const unsigned int indx ( ivec.size()==1 ? 0 : i ) ;

	DimVec::const_iterator dsrc ( dvec.begin() + ivec[indx]*nPerShape ) ;

	for( unsigned int j ( 0 ) ; j != nPerShape ; ++j )
	{
	    dims.push_back( *dsrc ) ;
	    ++dsrc ;
	}

	const CCGFloat* myParm ( CaloCellGeometry::getParmPtr( dims, 
							       ptr->parMgr(), 
							       ptr->parVecVec() ) ) ;


	const DetId id ( hgcalTopology->denseId2detId(dins[i]) );
    
	Pt3D  lRef ;
	Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
	hcg->localCorners( lc, &dims.front(), dins[i], lRef ) ;

	const Pt3D lBck ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7] ) ) ; // ctr rear  face in local
	const Pt3D lCor ( lc[0] ) ;

	//----------------------------------- create transform from 6 numbers ---
	const unsigned int jj ( dins[i]*nTrParm ) ;
	Tr3D tr ;
	const ROOT::Math::Translation3D tl ( tvec[jj], tvec[jj+1], tvec[jj+2] ) ;
	const ROOT::Math::EulerAngles ea (
	    6==nTrParm ?
	    ROOT::Math::EulerAngles( tvec[jj+3], tvec[jj+4], tvec[jj+5] ) :
	    ROOT::Math::EulerAngles() ) ;
	const ROOT::Math::Transform3D rt ( ea, tl ) ;
	double xx,xy,xz,dx,yx,yy,yz,dy,zx,zy,zz,dz;
	rt.GetComponents(xx,xy,xz,dx,yx,yy,yz,dy,zx,zy,zz,dz) ;
	tr = Tr3D( CLHEP::HepRep3x3( xx, xy, xz,
				     yx, yy, yz,
				     zx, zy, zz ), 
		   CLHEP::Hep3Vector(dx,dy,dz)     );

	const Pt3D        gRef ( lRef ) ;
	const GlobalPoint fCtr ( gRef.x(), gRef.y(), gRef.z() ) ;
	const Pt3D        gBck ( lBck ) ;
	const GlobalPoint fBck ( gBck.x(), gBck.y(), gBck.z() ) ;
	const Pt3D        gCor ( lCor ) ;
	const GlobalPoint fCor ( gCor.x(), gCor.y(), gCor.z() ) ;

	assert( hgcalTopology->detId2denseId(id) == dins[i] );

	ptr->newCell(  fCtr, fBck, fCor, myParm, id ) ;
    }

    ptr->initializeParms() ; // initializations; must happen after cells filled

    return ptr ; 
}

template class CaloGeometryDBEP< HGCalGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< HGCalGeometry , CaloGeometryDBReader> 
HGCalGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HGCalGeometryFromDBEP);

