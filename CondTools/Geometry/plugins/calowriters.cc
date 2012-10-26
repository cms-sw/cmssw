#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBWriter.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

template<>
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBWriter>::PtrType
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBWriter>::produceAligned( const typename HcalGeometry::AlignedRecord& iRecord ) 
{
    const Alignments* alignPtr  ( 0 ) ;
    const Alignments* globalPtr ( 0 ) ;
    if( m_applyAlignment ) // get ptr if necessary
    {
	edm::ESHandle< Alignments >                                      alignments ;
	iRecord.getRecord< typename HcalGeometry::AlignmentRecord >().get( alignments ) ;

	assert( alignments.isValid() && // require valid alignments and expected size
		( alignments->m_align.size() == HcalGeometry::numberOfAlignments() ) ) ;
	alignPtr = alignments.product() ;

	edm::ESHandle< Alignments >                          globals   ;
	iRecord.getRecord<GlobalPositionRcd>().get( globals ) ;

	assert( globals.isValid() ) ;
	globalPtr = globals.product() ;
    }

    TrVec  tvec ;
    DimVec dvec ;
    IVec   ivec ;

    if( CaloGeometryDBWriter::writeFlag() )
    {
	edm::ESHandle<CaloSubdetectorGeometry> pG ;
	iRecord.get( HcalGeometry::producerTag() + std::string("_master"), pG ) ; 

	const CaloSubdetectorGeometry* pGptr ( pG.product() ) ;

	pGptr->getSummary( tvec, ivec, dvec ) ;

	CaloGeometryDBWriter::write( tvec, dvec, ivec, HcalGeometry::dbString() ) ;
    }
    else
    {
	//std::cout<<"Getting Geometry from DB for "<<HcalGeometry::producerTag()<<std::endl ;
	edm::ESHandle<PCaloGeometry> pG ;
	iRecord.getRecord<typename HcalGeometry::PGeometryRecord >().get( pG ) ; 

	tvec = pG->getTranslation() ;
	dvec = pG->getDimension() ;
	ivec = pG->getIndexes() ;
    }	 
    //*********************************************************************************************

    const unsigned int nTrParm ( tvec.size()/HcalGeometry::k_NumberOfCellsForCorners ) ;

    edm::ESHandle<HcalTopology> hcalTopology;
    iRecord.getRecord<IdealGeometryRecord>().get( hcalTopology );

    assert( dvec.size() == hcalTopology->getNumberOfShapes() * HcalGeometry::k_NumberOfParametersPerShape ) ;
    PtrType ptr ( new HcalGeometry( &*hcalTopology ));

    ptr->fillDefaultNamedParameters() ;

    ptr->allocateCorners( HcalGeometry::k_NumberOfCellsForCorners ) ;

    ptr->allocatePar(    dvec.size() ,
			 HcalGeometry::k_NumberOfParametersPerShape ) ;

    for( unsigned int i ( 0 ) ; i != HcalGeometry::k_NumberOfCellsForCorners ; ++i )
    {
	const unsigned int nPerShape ( HcalGeometry::k_NumberOfParametersPerShape ) ;
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


	const DetId id ( HcalGeometry::DetIdType::detIdFromDenseIndex( i ) ) ;
    
	const unsigned int iGlob ( 0 == globalPtr ? 0 :
				   HcalGeometry::alignmentTransformIndexGlobal( id ) ) ;

	assert( 0 == globalPtr || iGlob < globalPtr->m_align.size() ) ;

	const AlignTransform* gt ( 0 == globalPtr ? 0 : &globalPtr->m_align[ iGlob ] ) ;

	assert( 0 == gt || iGlob == HcalGeometry::alignmentTransformIndexGlobal( DetId( gt->rawId() ) ) ) ;

	const unsigned int iLoc ( 0 == alignPtr ? 0 :
				  HcalGeometry::alignmentTransformIndexLocal( id ) ) ;

	assert( 0 == alignPtr || iLoc < alignPtr->m_align.size() ) ;

	const AlignTransform* at ( 0 == alignPtr ? 0 :
				   &alignPtr->m_align[ iLoc ] ) ;

	assert( 0 == at || ( HcalGeometry::alignmentTransformIndexLocal( DetId( at->rawId() ) ) == iLoc ) ) ;

	const CaloGenericDetId gId ( id ) ;

	Pt3D  lRef ;
	Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
	HcalGeometry::localCorners( lc, &dims.front(), i, lRef ) ;

	const Pt3D lBck ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7] ) ) ; // ctr rear  face in local
	const Pt3D lCor ( lc[0] ) ;

	//----------------------------------- create transform from 6 numbers ---
	const unsigned int jj ( i*nTrParm ) ;
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

	// now prepend alignment(s) for final transform
	const Tr3D atr ( 0 == at ? tr :
			 ( 0 == gt ? at->transform()*tr :
			   at->transform()*gt->transform()*tr ) ) ;
	//--------------------------------- done making transform  ---------------

	const Pt3D        gRef ( atr*lRef ) ;
	const GlobalPoint fCtr ( gRef.x(), gRef.y(), gRef.z() ) ;
	const Pt3D        gBck ( atr*lBck ) ;
	const GlobalPoint fBck ( gBck.x(), gBck.y(), gBck.z() ) ;
	const Pt3D        gCor ( atr*lCor ) ;
	const GlobalPoint fCor ( gCor.x(), gCor.y(), gCor.z() ) ;

	ptr->newCell(  fCtr, fBck, fCor, myParm, id ) ;
    }

    ptr->initializeParms() ; // initializations; must happen after cells filled

    return ptr ; 
}
 
template class CaloGeometryDBEP< EcalBarrelGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalEndcapGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBWriter> ;

template class CaloGeometryDBEP< HcalGeometry, CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< CaloTowerGeometry     , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< ZdcGeometry           , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< CastorGeometry        , CaloGeometryDBWriter> ;

typedef CaloGeometryDBEP< EcalBarrelGeometry , CaloGeometryDBWriter> 
EcalBarrelGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryToDBEP);

typedef CaloGeometryDBEP< EcalEndcapGeometry , CaloGeometryDBWriter> 
EcalEndcapGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryToDBEP);

typedef CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBWriter> 
EcalPreshowerGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryToDBEP);

typedef CaloGeometryDBEP< HcalGeometry , CaloGeometryDBWriter> 
HcalGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HcalGeometryToDBEP);

typedef CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBWriter> 
CaloTowerGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerGeometryToDBEP);

typedef CaloGeometryDBEP< ZdcGeometry , CaloGeometryDBWriter> 
ZdcGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(ZdcGeometryToDBEP);

typedef CaloGeometryDBEP< CastorGeometry , CaloGeometryDBWriter> 
CastorGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CastorGeometryToDBEP);
