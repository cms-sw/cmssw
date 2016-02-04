#ifndef GEOMETRY_ECALTBHODOSCOPEGEOMETRYLOADERFROMDDD_H
#define GEOMETRY_ECALTBHODOSCOPEGEOMETRYLOADERFROMDDD_H 1

#include "Geometry/EcalTestBeam/interface/EcalHodoscopeNumberingScheme.h"


/** \class EcalTBHodoscopeGeometryLoaderFromDDD
 *
 *
 *   
 * $Id: EcalTBHodoscopeGeometryLoaderFromDDD.h,v 1.3 2011/06/03 19:20:55 heltsley Exp $
 * \author P. Meridiani - INFN Roma 1
*/

class DDCompactView;
class DDFilteredView;
class DDFilter;
class CaloSubdetectorGeometry;
class EcalTBHodoscopeGeometry;

#include <memory>
#include <string>

class EcalTBHodoscopeGeometryLoaderFromDDD
{
   public:

      EcalTBHodoscopeGeometryLoaderFromDDD() {} ;

      virtual ~EcalTBHodoscopeGeometryLoaderFromDDD() {} ;

      std::auto_ptr<CaloSubdetectorGeometry> load( const DDCompactView* cpv ) ;

   private:

      void makeGeometry( const DDCompactView*     cpv ,
			 CaloSubdetectorGeometry* ebg  ) ;

      unsigned int getDetIdForDDDNode( const DDFilteredView &fv ) ;

      std::string getDDDString( std::string s, DDFilteredView* fv ) ;

      DDFilter* getDDFilter();

      EcalHodoscopeNumberingScheme _scheme;
};

#endif
