#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H 1

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondFormats/Alignment/interface/Alignments.h"

//Forward declaration

//
// class declaration
//

template <class T>
class CaloGeometryEPtest : public edm::ESProducer 
{
   public:

      typedef CaloGeometryLoaderTest<T>          LoaderType ;
      typedef typename LoaderType::PtrType   PtrType    ;

      CaloGeometryEPtest<T>( const edm::ParameterSet& ps ) :
	 m_applyAlignment ( ps.getParameter<bool>("applyAlignment") )
      {
	 setWhatProduced( this,
			  &CaloGeometryEPtest<T>::produceAligned,
			  edm::es::Label( T::producerTag() ) ) ;
      }

      virtual ~CaloGeometryEPtest<T>() {}
      PtrType produceAligned( const typename T::AlignedRecord& iRecord ) 
      {
	 const Alignments* alignPtr  ( 0 ) ;
	 const Alignments* globalPtr ( 0 ) ;
	 if( m_applyAlignment ) // get ptr if necessary
	 {
	    edm::ESHandle< Alignments >                                      alignments ;
	    iRecord.template getRecord< typename T::AlignmentRecord >().get( alignments ) ;

	    assert( alignments.isValid() && // require valid alignments and expected size
		    ( alignments->m_align.size() == T::numberOfAlignments() ) ) ;
	    alignPtr = alignments.product() ;

	    edm::ESHandle< Alignments >                          globals   ;
	    iRecord.template getRecord<GlobalPositionRcd>().get( globals ) ;

	    assert( globals.isValid() ) ;
	    globalPtr = globals.product() ;
	 }
	 edm::ESHandle< DDCompactView > cpv ;
	 iRecord.template getRecord<IdealGeometryRecord>().get( cpv ) ;

	 LoaderType loader ;
	 PtrType ptr ( loader.load( &(*cpv), alignPtr, globalPtr ) ) ; // no temporaries for shared+ptr!! 

	 return ptr ; 
      }

   private:


      bool        m_applyAlignment ;
};

#endif
