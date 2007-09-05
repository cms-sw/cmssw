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
#include "Geometry/CaloGeometry/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

//Forward declaration

//
// class declaration
//

template <class T>
class CaloGeometryEP : public edm::ESProducer 
{
   public:

      typedef CaloGeometryLoader<T> LoaderType ;

      CaloGeometryEP<T>( const edm::ParameterSet& ps )  :
	 m_loader ( new LoaderType ) 
      {
	 setWhatProduced( this, T::producerName() ) ;
      }

      virtual ~CaloGeometryEP<T>() { delete m_loader ; }

      typename LoaderType::PtrType produce( const IdealGeometryRecord& iRecord ) 
      {
	 using namespace edm::es;

	 edm::ESHandle< DDCompactView > cpv ;
	 iRecord.get( cpv ) ;

	 return m_loader->load( &( *cpv ) ) ;
      }

   private:

      LoaderType* m_loader ;
};

#endif
