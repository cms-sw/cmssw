#ifndef GEOMETRY_ECALGEOMETRYLOADER_H
#define GEOMETRY_ECALGEOMETRYLOADER_H 1

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

#include "CondFormats/Alignment/interface/Alignments.h"

#include "CLHEP/Geometry/Transform3D.h"
#include <string>
#include <vector>

/** \class CaloGeometryLoader<T>
 *
 * Templated class for calo subdetector geometry loaders from DDD.
*/

class DDCompactView;

template < class T >
class CaloGeometryLoader
{
public:

  typedef std::vector< double > ParmVec ;

  typedef std::shared_ptr<CaloSubdetectorGeometry > PtrType ;

  typedef CaloSubdetectorGeometry::ParVec    ParVec ;
  typedef CaloSubdetectorGeometry::ParVecVec ParVecVec ;

  static const double k_ScaleFromDDDtoGeant ;

  CaloGeometryLoader< T >() ;

  virtual ~CaloGeometryLoader< T >() {}
 
  PtrType load( const DDCompactView* cpv,
		const Alignments*    alignments = nullptr ,
		const Alignments*    globals    = nullptr  ) ;  

private:

  void makeGeometry( const DDCompactView*  cpv        , 
		     T*                    geom       ,
		     const Alignments*     alignments ,
		     const Alignments*     globals       ) ;
      
  void fillNamedParams( const DDFilteredView& fv,
			T*             geom ) ;
      
  void fillGeom( T*                    geom ,
		 const ParmVec&        pv ,
		 const HepGeom::Transform3D& tr ,
		 const DetId&          id    ) ;

  unsigned int getDetIdForDDDNode( const DDFilteredView& fv ) ;

  typename T::NumberingScheme m_scheme;
  DDAndFilter<DDSpecificsMatchesValueFilter,
              DDSpecificsMatchesValueFilter> m_filter;
};

#endif
