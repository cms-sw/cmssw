#ifndef Surface_GeometricSorting_h
#define Surface_GeometricSorting_h

#include <functional>
#include <DataFormats/GeometryVector/interface/Phi.h>

namespace geomsort{

/** \class ExtractR
 *
 *  functor to sort in R using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractR<Surface>());
 *
 *  $Date: 2007/01/17 20:58:43 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  template <class T, class Scalar = typename T::Scalar>
  struct ExtractR {
    typedef Scalar result_type;
    Scalar operator()(const T* p) const {return p->position().perp();}
    Scalar operator()(const T& p) const {return p.position().perp();}
  };


/** \class ExtractPhi
 *
 *  functor to sort in phi (from -pi to pi) using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *
 *  Note that sorting in phi is done within the phi range of 
 *  (-pi, pi]. It may NOT be what you expect if the elements cluster around
 *  the pi discontinuity.
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractPhi<Surface>());
 *
 *  $Date: 2007/01/17 20:58:43 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  template <class T, class Scalar = typename T::Scalar>
  struct ExtractPhi {
    typedef Geom::Phi<Scalar> result_type;
    Geom::Phi<Scalar> operator()(const T* p) const {return p->position().phi();}
    Geom::Phi<Scalar> operator()(const T& p) const {return p.position().phi();}
  };



/** \class ExtractZ
 *
 *  functor to sort in Z using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractZ<Surface>());
 *
 *  $Date: 2007/01/17 20:58:43 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  template <class T, class Scalar = typename T::Scalar>
  struct ExtractZ {
    typedef Scalar result_type;
    Scalar operator()(const T* p) const {return p->position().z();}
    Scalar operator()(const T& p) const {return p.position().z();}
  };

/** \class ExtractAbsZ
 *
 *  functor to sort in |Z| using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractAbsZ<Surface>());
 *
 *  $Date: 2007/01/17 20:58:43 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

  template <class T, class Scalar = typename T::Scalar>
  struct ExtractAbsZ {
    typedef Scalar result_type;
    Scalar operator()(const T* p) const {return fabs(p->position().z());}
    Scalar operator()(const T& p) const {return fabs(p.position().z());}
  };


}
#endif

