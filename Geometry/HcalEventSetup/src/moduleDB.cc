#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

template<>
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBReader>::produceAligned( const typename HcalGeometry::AlignedRecord& iRecord ) {
  const Alignments* alignPtr  ( nullptr ) ;
  const Alignments* globalPtr ( nullptr ) ;
  if( m_applyAlignment ) {// get ptr if necessary
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
  IVec   dins ;

  if( CaloGeometryDBReader::writeFlag() ) {
    edm::ESHandle<CaloSubdetectorGeometry> pG ;
    iRecord.get( HcalGeometry::producerTag() + std::string("_master"), pG ) ; 

    const CaloSubdetectorGeometry* pGptr ( pG.product() ) ;

    pGptr->getSummary( tvec, ivec, dvec, dins ) ;

    CaloGeometryDBReader::writeIndexed( tvec, dvec, ivec, dins, HcalGeometry::dbString() ) ;
  } else {
    edm::ESHandle<PCaloGeometry> pG ;
    iRecord.getRecord<typename HcalGeometry::PGeometryRecord >().get( pG ) ; 

    tvec = pG->getTranslation() ;
    dvec = pG->getDimension() ;
    ivec = pG->getIndexes() ;
    dins = pG->getDenseIndices();
  }	 
  //*********************************************************************************************

  edm::ESHandle<HcalTopology> hcalTopology;
  iRecord.getRecord<HcalRecNumberingRecord>().get( hcalTopology );

  // We know that the numer of shapes chanes with changing depth
  // so, this check is temporary disabled. We need to implement
  // a way either to store or calculate the number of shapes or be able
  // to deal with only max numer of shapes.
  // assert( dvec.size() == hcalTopology->getNumberOfShapes() * HcalGeometry::k_NumberOfParametersPerShape ) ;
  assert( dvec.size() <= hcalTopology->getNumberOfShapes() * HcalGeometry::k_NumberOfParametersPerShape ) ;
  HcalGeometry* hcg = new HcalGeometry( *hcalTopology );
    
  PtrType ptr ( hcg );
 
  const unsigned int nTrParm( hcg->numberOfTransformParms());
   
  ptr->fillDefaultNamedParameters();
  ptr->allocateCorners( hcalTopology->ncells()+hcalTopology->getHFSize());
  ptr->allocatePar( dvec.size() ,
		    HcalGeometry::k_NumberOfParametersPerShape );

  for( unsigned int i ( 0 ) ; i < dins.size() ; ++i ) {
    const unsigned int nPerShape ( HcalGeometry::k_NumberOfParametersPerShape );
    DimVec dims;
    dims.reserve( nPerShape );

    const unsigned int indx( ivec.size() == 1 ? 0 : i );

    DimVec::const_iterator dsrc( dvec.begin() + ivec[indx]*nPerShape );

    for( unsigned int j ( 0 ) ; j != nPerShape ; ++j ) {
      dims.emplace_back( *dsrc ) ;
      ++dsrc ;
    }

    const CCGFloat* myParm( CaloCellGeometry::getParmPtr( dims, 
							  ptr->parMgr(), 
							  ptr->parVecVec()));

    const DetId id( hcalTopology->denseId2detId( dins[i]));
    
    const unsigned int iGlob( nullptr == globalPtr ? 0 :
			      HcalGeometry::alignmentTransformIndexGlobal( id ));

    assert( nullptr == globalPtr || iGlob < globalPtr->m_align.size());

    const AlignTransform* gt ( nullptr == globalPtr ? nullptr : &globalPtr->m_align[ iGlob ] );

    assert( nullptr == gt || iGlob == HcalGeometry::alignmentTransformIndexGlobal( DetId( gt->rawId())));

    const unsigned int iLoc( nullptr == alignPtr ? 0 :
			     HcalGeometry::alignmentTransformIndexLocal( id ));

    assert( nullptr == alignPtr || iLoc < alignPtr->m_align.size());

    const AlignTransform* at( nullptr == alignPtr ? nullptr :
			      &alignPtr->m_align[ iLoc ]);

    assert( nullptr == at || ( HcalGeometry::alignmentTransformIndexLocal( DetId( at->rawId())) == iLoc ));

    Pt3D  lRef ;
    Pt3DVec lc( 8, Pt3D( 0, 0, 0 ));
    hcg->localCorners( lc, &dims.front(), dins[i], lRef );

    const Pt3D lBck( 0.25*(lc[4]+lc[5]+lc[6]+lc[7] )); // ctr rear  face in local
    const Pt3D lCor( lc[0] );

    //----------------------------------- create transform from 6 numbers ---
    const unsigned int jj( i * nTrParm ); // Note: Dence indices are not sorted and
                                          // parameters stored according to order of a cell creation
    Tr3D tr;
    const ROOT::Math::Translation3D tl( tvec[jj], tvec[jj+1], tvec[jj+2] );
    const ROOT::Math::EulerAngles ea( 6 == nTrParm ?
				      ROOT::Math::EulerAngles( tvec[jj+3], tvec[jj+4], tvec[jj+5] ) :
				      ROOT::Math::EulerAngles());
    const ROOT::Math::Transform3D rt( ea, tl );
    double xx, xy, xz, dx;
    double yx, yy, yz, dy;
    double zx, zy, zz, dz;
    rt.GetComponents( xx, xy, xz, dx,
		      yx, yy, yz, dy,
		      zx, zy, zz, dz );
    tr = Tr3D( CLHEP::HepRep3x3( xx, xy, xz,
				 yx, yy, yz,
				 zx, zy, zz ), 
	       CLHEP::Hep3Vector( dx, dy, dz));

    // now prepend alignment(s) for final transform
    const Tr3D atr( nullptr == at ? tr :
		    ( nullptr == gt ? at->transform() * tr :
		      at->transform() * gt->transform() * tr ));
    //--------------------------------- done making transform  ---------------

    const Pt3D        gRef( atr*lRef ) ;
    const GlobalPoint fCtr( gRef.x(), gRef.y(), gRef.z() ) ;
    const Pt3D        gBck( atr*lBck ) ;
    const GlobalPoint fBck( gBck.x(), gBck.y(), gBck.z() ) ;
    const Pt3D        gCor( atr*lCor ) ;
    const GlobalPoint fCor( gCor.x(), gCor.y(), gCor.z() ) ;

    assert( hcalTopology->detId2denseId(id) == dins[i] );

    ptr->newCell( fCtr, fBck, fCor, myParm, id ) ;
  }

  ptr->initializeParms(); // initializations; must happen after cells filled

  return ptr; 
}

template<>
CaloGeometryDBEP<CaloTowerGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<CaloTowerGeometry, CaloGeometryDBReader>::produceAligned( const typename CaloTowerGeometry::AlignedRecord& iRecord ) {

  const Alignments* alignPtr  ( nullptr ) ;
  const Alignments* globalPtr ( nullptr ) ;
  if( m_applyAlignment ) { // get ptr if necessary
    edm::ESHandle< Alignments >                                      alignments ;
    iRecord.getRecord< typename CaloTowerGeometry::AlignmentRecord >().get( alignments ) ;

    assert( alignments.isValid() && // require valid alignments and expected sizet
	    ( alignments->m_align.size() == CaloTowerGeometry::numberOfAlignments() ) ) ;
    alignPtr = alignments.product() ;

    edm::ESHandle< Alignments >                          globals   ;
    iRecord.getRecord<GlobalPositionRcd>().get( globals ) ;

    assert( globals.isValid() ) ;
    globalPtr = globals.product() ;
  }

  TrVec  tvec ;
  DimVec dvec ;
  IVec   ivec ;
  IVec   dins ;
  
  if( CaloGeometryDBReader::writeFlag() ) {
    edm::ESHandle<CaloSubdetectorGeometry> pG ;
    iRecord.get( CaloTowerGeometry::producerTag() + std::string("_master"), pG ) ; 

    const CaloSubdetectorGeometry* pGptr ( pG.product() ) ;

    pGptr->getSummary( tvec, ivec, dvec, dins ) ;
    
    CaloGeometryDBReader::writeIndexed( tvec, dvec, ivec, dins, CaloTowerGeometry::dbString() ) ;
  } else {
    edm::ESHandle<PCaloGeometry> pG ;
    iRecord.getRecord<typename CaloTowerGeometry::PGeometryRecord >().get( pG ) ; 

    tvec = pG->getTranslation() ;
    dvec = pG->getDimension() ;
    ivec = pG->getIndexes() ;
    dins = pG->getDenseIndices();
  }	 
//*********************************************************************************************

  edm::ESHandle<CaloTowerTopology> caloTopology;
  iRecord.getRecord<HcalRecNumberingRecord>().get( caloTopology );


  CaloTowerGeometry* ctg=new CaloTowerGeometry( &*caloTopology );

  const unsigned int nTrParm ( tvec.size()/ctg->numberOfCellsForCorners() ) ;

  assert( dvec.size() == ctg->numberOfShapes() * CaloTowerGeometry::k_NumberOfParametersPerShape ) ;


  PtrType ptr ( ctg ) ;

  ptr->fillDefaultNamedParameters() ;

  ptr->allocateCorners( ctg->numberOfCellsForCorners() ) ;

  ptr->allocatePar(    dvec.size() ,
		       CaloTowerGeometry::k_NumberOfParametersPerShape ) ;

  for( unsigned int i ( 0 ) ;  i < dins.size() ; ++i ) {
    const unsigned int nPerShape ( ctg->numberOfParametersPerShape() ) ;
    DimVec dims ;
    dims.reserve( nPerShape ) ;
    
    const unsigned int indx ( ivec.size()==1 ? 0 : i ) ;
	    
    DimVec::const_iterator dsrc ( dvec.begin() + ivec[indx]*nPerShape ) ;

    for( unsigned int j ( 0 ) ; j != nPerShape ; ++j )  {
      dims.emplace_back( *dsrc ) ;
      ++dsrc ;
    }

    const CCGFloat* myParm ( CaloCellGeometry::getParmPtr( dims, 
							   ptr->parMgr(), 
							   ptr->parVecVec() ));


    const DetId id ( caloTopology->detIdFromDenseIndex(dins[i]) ) ;
    
    const unsigned int iGlob ( nullptr == globalPtr ? 0 :
			       ctg->alignmentTransformIndexGlobal( id ) ) ;

    assert( nullptr == globalPtr || iGlob < globalPtr->m_align.size() ) ;

    const AlignTransform* gt ( nullptr == globalPtr ? nullptr : &globalPtr->m_align[ iGlob ] ) ;

    assert( nullptr == gt || iGlob == ctg->alignmentTransformIndexGlobal( DetId( gt->rawId() ) ) ) ;

    const unsigned int iLoc ( nullptr == alignPtr ? 0 :
			      ctg->alignmentTransformIndexLocal( id ) ) ;

    assert( nullptr == alignPtr || iLoc < alignPtr->m_align.size() ) ;

    const AlignTransform* at ( nullptr == alignPtr ? nullptr :
			       &alignPtr->m_align[ iLoc ] ) ;

    assert( nullptr == at || ( ctg->alignmentTransformIndexLocal( DetId( at->rawId() ) ) == iLoc ) ) ;

    const CaloGenericDetId gId ( id ) ;

    Pt3D  lRef ;
    Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
    ctg->localCorners( lc, &dims.front(), dins[i], lRef ) ;

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
    const Tr3D atr ( nullptr == at ? tr :
		     ( nullptr == gt ? at->transform()*tr :
		       at->transform()*gt->transform()*tr ) ) ;
    //--------------------------------- done making transform  ---------------

    const Pt3D        gRef ( atr*lRef ) ;
    const GlobalPoint fCtr ( gRef.x(), gRef.y(), gRef.z() ) ;
    const Pt3D        gBck ( atr*lBck ) ;
    const GlobalPoint fBck ( gBck.x(), gBck.y(), gBck.z() ) ;
    const Pt3D        gCor ( atr*lCor ) ;
    const GlobalPoint fCor ( gCor.x(), gCor.y(), gCor.z() ) ;

    assert( caloTopology->denseIndex(id) == dins[i] );

    ptr->newCell(  fCtr, fBck, fCor, myParm, id ) ;
  }

  ptr->initializeParms() ; // initializations; must happen after cells filled

  return ptr ; 
}

template class CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> 
HcalGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HcalGeometryFromDBEP);

template class CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBReader> 
CaloTowerGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerGeometryFromDBEP);
