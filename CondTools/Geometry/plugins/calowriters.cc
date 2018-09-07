#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBWriter.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

template<>
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBWriter>::PtrType
CaloGeometryDBEP<HcalGeometry, CaloGeometryDBWriter>::produceAligned( const typename HcalGeometry::AlignedRecord& iRecord ) 
{
    const Alignments* alignPtr( nullptr );
    const Alignments* globalPtr( nullptr );
    if( m_applyAlignment ) // get ptr if necessary
    {
      edm::ESHandle< Alignments > alignments;
      iRecord.getRecord< typename HcalGeometry::AlignmentRecord >().get( alignments );
      
      assert( alignments.isValid() && // require valid alignments and expected size
	      ( alignments->m_align.size() == HcalGeometry::numberOfAlignments()));
      alignPtr = alignments.product();
      
      edm::ESHandle< Alignments > globals;
      iRecord.getRecord<GlobalPositionRcd>().get( globals );
      
      assert( globals.isValid());
      globalPtr = globals.product();
    }

    TrVec  tvec;
    DimVec dvec;
    IVec   ivec;
    IVec   dins;

    if( CaloGeometryDBWriter::writeFlag())
    {
      edm::ESHandle<CaloSubdetectorGeometry> pG;
      iRecord.get( HcalGeometry::producerTag() + std::string("_master"), pG ); 

      const CaloSubdetectorGeometry* pGptr(pG.product());
      
      pGptr->getSummary( tvec, ivec, dvec, dins );
	
      CaloGeometryDBWriter::writeIndexed( tvec, dvec, ivec, dins, HcalGeometry::dbString());
    }
    else
    {
      edm::ESHandle<PCaloGeometry> pG;
      iRecord.getRecord<typename HcalGeometry::PGeometryRecord >().get( pG ); 

      tvec = pG->getTranslation();
      dvec = pG->getDimension();
      ivec = pG->getIndexes();
      dins = pG->getDenseIndices();
    }	 
    //*********************************************************************************************

    edm::ESHandle<HcalTopology> hcalTopology;
    iRecord.getRecord<HcalRecNumberingRecord>().get( hcalTopology );

    // We know that the numer of shapes chanes with changing depth
    // so, this check is temporary disabled. We need to implement
    // a way either to store or calculate the number of shapes or be able
    // to deal with only max numer of shapes.
    assert( dvec.size() <= hcalTopology->getNumberOfShapes() * HcalGeometry::k_NumberOfParametersPerShape );
    HcalGeometry* hcalGeometry = new HcalGeometry( *hcalTopology );
    PtrType ptr( hcalGeometry );

    const unsigned int nTrParm( hcalGeometry->numberOfTransformParms());
    
    ptr->fillDefaultNamedParameters();
    ptr->allocateCorners( hcalTopology->ncells() + hcalTopology->getHFSize());
    ptr->allocatePar( hcalGeometry->numberOfShapes(),
		      HcalGeometry::k_NumberOfParametersPerShape );

    for( unsigned int i ( 0 ) ; i < dins.size(); ++i )
    {
      const unsigned int nPerShape( HcalGeometry::k_NumberOfParametersPerShape );
      DimVec dims;
      dims.reserve( nPerShape );

      const unsigned int indx( ivec.size() == 1 ? 0 : i );

      DimVec::const_iterator dsrc( dvec.begin() + ivec[indx] * nPerShape );

      for( unsigned int j ( 0 ) ; j != nPerShape ; ++j )
      {
	dims.push_back( *dsrc );
	++dsrc ;
      }

      const CCGFloat* myParm ( CaloCellGeometry::getParmPtr( dims, 
							     ptr->parMgr(), 
							     ptr->parVecVec()));
	
      const DetId id( hcalTopology->denseId2detId( dins[i]));
	
      const unsigned int iGlob( nullptr == globalPtr ? 0 :
				HcalGeometry::alignmentTransformIndexGlobal( id ));

      assert( nullptr == globalPtr || iGlob < globalPtr->m_align.size());

      const AlignTransform* gt( nullptr == globalPtr ? nullptr : &globalPtr->m_align[ iGlob ]);

      assert( nullptr == gt || iGlob == HcalGeometry::alignmentTransformIndexGlobal( DetId( gt->rawId())));

      const unsigned int iLoc( nullptr == alignPtr ? 0 :
			       HcalGeometry::alignmentTransformIndexLocal( id ));

      assert( nullptr == alignPtr || iLoc < alignPtr->m_align.size());

      const AlignTransform* at( nullptr == alignPtr ? nullptr :
				&alignPtr->m_align[ iLoc ]);

      assert( nullptr == at || ( HcalGeometry::alignmentTransformIndexLocal( DetId( at->rawId())) == iLoc ));

      Pt3D  lRef;
      Pt3DVec lc( 8, Pt3D( 0, 0, 0 ));
      hcalGeometry->localCorners( lc, &dims.front(), dins[i], lRef );

      const Pt3D lBck( 0.25*(lc[4]+lc[5]+lc[6]+lc[7] )); // ctr rear  face in local
      const Pt3D lCor( lc[0] ) ;

      //----------------------------------- create transform from 6 numbers ---
      const unsigned int jj( i * nTrParm );
	
      Tr3D tr;
      const ROOT::Math::Translation3D tl( tvec[jj], tvec[jj+1], tvec[jj+2] );
      const ROOT::Math::EulerAngles ea( 6 == nTrParm ?
					ROOT::Math::EulerAngles( tvec[jj+3], tvec[jj+4], tvec[jj+5] ) :
					ROOT::Math::EulerAngles());
      const ROOT::Math::Transform3D rt( ea, tl );
      double xx, xy, xz, dx;
      double yx, yy, yz, dy;
      double zx, zy, zz, dz;
      rt.GetComponents(xx,xy,xz,dx,yx,yy,yz,dy,zx,zy,zz,dz) ;
      tr = Tr3D( CLHEP::HepRep3x3( xx, xy, xz,
				   yx, yy, yz,
				   zx, zy, zz ), 
		 CLHEP::Hep3Vector( dx, dy, dz ));

      // now prepend alignment(s) for final transform
      const Tr3D atr( nullptr == at ? tr :
		      ( nullptr == gt ? at->transform() * tr :
			at->transform() * gt->transform() * tr ));
      //--------------------------------- done making transform  ---------------

      const Pt3D gRef( atr*lRef );
      const GlobalPoint fCtr( gRef.x(), gRef.y(), gRef.z());
      const Pt3D        gBck( atr*lBck );
      const GlobalPoint fBck( gBck.x(), gBck.y(), gBck.z());
      const Pt3D        gCor( atr*lCor );
      const GlobalPoint fCor( gCor.x(), gCor.y(), gCor.z());

      assert( hcalTopology->detId2denseId(id) == dins[i] );
      ptr->newCell( fCtr, fBck, fCor, myParm, id );
    }
    
    ptr->initializeParms(); // initializations; must happen after cells filled
    
    return ptr; 
}

template<>
CaloGeometryDBEP<CaloTowerGeometry, CaloGeometryDBWriter>::PtrType
CaloGeometryDBEP<CaloTowerGeometry, CaloGeometryDBWriter>::produceAligned( const typename CaloTowerGeometry::AlignedRecord& iRecord ) {

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
  
  if( CaloGeometryDBWriter::writeFlag() ) {
    edm::ESHandle<CaloSubdetectorGeometry> pG ;
    iRecord.get( CaloTowerGeometry::producerTag() + std::string("_master"), pG ) ; 

    const CaloSubdetectorGeometry* pGptr(pG.product());

    pGptr->getSummary( tvec, ivec, dvec, dins ) ;
    
    CaloGeometryDBWriter::writeIndexed( tvec, dvec, ivec, dins, CaloTowerGeometry::dbString() ) ;
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
      dims.push_back( *dsrc ) ;
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

template<>
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBWriter>::PtrType
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBWriter>::produceAligned( const typename HGCalGeometry::AlignedRecord& iRecord ) 
{
  TrVec  tvec; // transformation
  DimVec dvec; // parameters
  IVec   ivec; // layers
  IVec   dins; // valid geom ids

  std::string name;

  name = "HGCalEESensitive";
  
  if( CaloGeometryDBWriter::writeFlag() )
  {
    edm::ESHandle<HGCalGeometry> geomH;
    iRecord.getRecord<IdealGeometryRecord>().get( name, geomH );
    const HGCalGeometry* geom = geomH.product();
    
    geom->getSummary( tvec, ivec, dvec, dins ) ;
	
    CaloGeometryDBWriter::writeIndexed( tvec, dvec, ivec, dins, HGCalGeometry::dbString() ) ;
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
      dims.push_back( *dsrc ) ;
      ++dsrc ;
    }
    
    std::vector<GlobalPoint> corners( FlatHexagon::ncorner_ );
    
    FlatHexagon::createCorners( dims, tr, corners );
    
    const CCGFloat* myParm( CaloCellGeometry::getParmPtr( dims, 
							  ptr->parMgr(), 
							  ptr->parVecVec()));
    GlobalPoint front ( FlatHexagon::oneBySix_*( corners[0].x() + 
						 corners[1].x() + 
						 corners[2].x() + 
						 corners[3].x() + 
						 corners[4].x() + 
						 corners[5].x()),
			FlatHexagon::oneBySix_*( corners[0].y() + 
						 corners[1].y() + 
						 corners[2].y() + 
						 corners[3].y() + 
						 corners[4].y() + 
						 corners[5].y()),
			FlatHexagon::oneBySix_*( corners[0].z() + 
						 corners[1].z() + 
						 corners[2].z() + 
						 corners[3].z() + 
						 corners[4].z() + 
						 corners[5].z()));
    
    GlobalPoint back  ( FlatHexagon::oneBySix_*( corners[6].x() + 
						 corners[7].x() + 
						 corners[8].x() + 
						 corners[9].x() + 
						 corners[10].x()+ 
						 corners[11].x()),
			FlatHexagon::oneBySix_*( corners[6].y() + 
						 corners[7].y() + 
						 corners[8].y() + 
						 corners[9].y() + 
						 corners[10].y()+ 
						 corners[11].y()),
			FlatHexagon::oneBySix_*( corners[6].z() + 
						 corners[7].z() + 
						 corners[8].z() + 
						 corners[9].z() + 
						 corners[10].z() + 
						 corners[11].z()));
    
    if (front.mag2() > back.mag2()) { // front should always point to the center, so swap front and back
      std::swap (front, back);
      std::swap_ranges (corners.begin(), 
			corners.begin()+FlatHexagon::ncornerBy2_, 
			corners.begin()+FlatHexagon::ncornerBy2_); 
    }
    
    ptr->newCell( front, back, corners[0], myParm, id );
  }
  
  ptr->initializeParms(); // initializations; must happen after cells filled
  
  return ptr;
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

typedef CaloGeometryDBEP< HGCalGeometry , CaloGeometryDBWriter> 
HGCalGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HGCalGeometryToDBEP);
