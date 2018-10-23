#ifndef GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H
#define GEOMETRY_CALOGEOMETRY_CALOGEOMETRYEP_H 1

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
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

      using LoaderType = CaloGeometryLoader<T>;
      using PtrType = typename LoaderType::PtrType;

      CaloGeometryEP<T>( const edm::ParameterSet& ps ) :
	 m_applyAlignment ( ps.getParameter<bool>("applyAlignment") )
      {
	 setWhatProduced( this,
			  &CaloGeometryEP<T>::produceAligned,
//			  dependsOn( &CaloGeometryEP<T>::idealRecordCallBack ),
			  edm::es::Label( T::producerTag() ) ) ;
      }

      ~CaloGeometryEP<T>() override {}
      PtrType produceAligned( const typename T::AlignedRecord& iRecord ) 
      {
	 const Alignments* alignPtr  ( nullptr ) ;
	 const Alignments* globalPtr ( nullptr ) ;
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
	 edm::ESTransientHandle<DDCompactView> cpv ;
	 iRecord.template getRecord<IdealGeometryRecord>().get( cpv ) ;

	 LoaderType loader ;
	 return loader.load( &(*cpv), alignPtr, globalPtr );
      }

   private:


      bool        m_applyAlignment ;
};

#endif
