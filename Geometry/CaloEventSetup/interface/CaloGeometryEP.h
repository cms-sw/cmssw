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
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "CondFormats/Alignment/interface/Alignments.h"

//Forward declaration

//
// class declaration
//

template <class T>
class CaloGeometryEP : public edm::ESProducer 
{
   public:

      typedef CaloGeometryLoader<T>          LoaderType ;
      typedef typename LoaderType::PtrType   PtrType    ;

      CaloGeometryEP<T>( const edm::ParameterSet& ps ) :
	 m_cpv    ( 0 ) ,
	 m_applyAlignment ( ps.getUntrackedParameter<bool>("applyAlignment", false) )
      {
	 setWhatProduced( this,
			  &CaloGeometryEP<T>::produceAligned,
			  dependsOn( &CaloGeometryEP<T>::idealRecordCallBack ),
			  edm::es::Label( T::producerName() ) ) ;

// disable
//	 setWhatProduced( this, 
//			  &CaloGeometryEP<T>::produceIdeal,
//			  edm::es::Label( T::producerName() ) ) ;
      }

      virtual ~CaloGeometryEP<T>() {}
      PtrType produceAligned( const typename T::AlignedRecord& iRecord ) 
      {
	 assert( 0 != m_cpv ) ;     // should have been filled by call to callback method below

	 const Alignments* alignPtr ( 0 ) ;
	 if( m_applyAlignment ) // get ptr if necessary
	 {
	    edm::ESHandle< Alignments >                                      alignments ;
	    iRecord.template getRecord< typename T::AlignmentRecord >().get( alignments ) ;

	    assert( alignments.isValid() && // require valid alignments and expected size
		    ( alignments->m_align.size() == T::numberOfAlignments() ) ) ;
	    alignPtr = alignments.product() ;
	 }
	 LoaderType loader ;
	 PtrType ptr ( loader.load( m_cpv, alignPtr ) ) ; // no temporaries for shared+ptr!! 
	 return ptr ; 
      }

      PtrType produceIdeal(     const typename T::IdealRecord& iRecord )
      {
	 assert( !m_applyAlignment ) ;
	 idealRecordCallBack( iRecord ) ; // must call manually because is same record
	 assert( 0 != m_cpv ) ;
	 LoaderType loader ;
	 PtrType ptr ( loader.load( m_cpv ) ) ; // no temporaries for shared+ptr!! 
	 return ptr ; 
      }

      void idealRecordCallBack( const typename T::IdealRecord& iRecord )
      {
	 edm::ESHandle< DDCompactView > cpv ;
	 iRecord.get( cpv ) ;

	 m_cpv = &( *cpv ) ;
      }

   private:

      const DDCompactView* m_cpv ;

      bool        m_applyAlignment ;
};

#endif
