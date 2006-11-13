#ifndef DetSorting_h
#define DetSorting_h

#include <Geometry/Surface/interface/GeometricSorting.h>
#include "Geometry/CommonDetUnit/interface/GeomDet.h"


namespace geomsort{

/** \typedef DetR
 *
 *  functor to sort in R using precomputed_value_sort.
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), DetR());
 *
 *  $Date: 2006/03/17 15:02:05 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  typedef ExtractR<GeomDet,float> DetR;


/** \typedef DetPhi
 *
 *  functor to sort in phi (from -pi to pi) using precomputed_value_sort.
 *
 *  Note that sorting in phi is done within the phi range of 
 *  (-pi, pi]. It may NOT be what you expect if the elements cluster around
 *  the pi discontinuity.
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), DetPhi());
 *
 *  $Date: 2006/03/17 15:02:05 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  typedef ExtractPhi<GeomDet,float> DetPhi;


/** \typedef DetZ
 *
 *  functor to sort in Z using precomputed_value_sort.
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), DetZ());
 *
 *  $Date: 2006/03/17 15:02:05 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  typedef ExtractZ<GeomDet,float> DetZ;


}
#endif


