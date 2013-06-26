#ifndef MagneticFieldProvider_h
#define MagneticFieldProvider_h

/** \class MagneticFieldProvider
 *
 *  Virtual interface for the field provider for an individual field volume.
 *
 *  $Date: 2011/07/29 14:57:40 $
 *  $Revision: 1.4 $
 *  \author T. Todorov
 */

#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"

template <class T>
class MagneticFieldProvider {
public:

  typedef Point3DBase<T,GlobalTag>      GlobalPointType;
  typedef Point3DBase<T,LocalTag>       LocalPointType;
  typedef Vector3DBase<T,GlobalTag>     GlobalVectorType;
  typedef Vector3DBase<T,LocalTag>      LocalVectorType;


  virtual ~MagneticFieldProvider(){}

  /** Returns the field vector in the local frame, at local position p
   */
  virtual LocalVectorType valueInTesla( const LocalPointType& p) const = 0;

  /** Returns the field vector in the global frame, at global position p
   * Not needed, the MagVolume does the transformation to global!
   */
  // virtual GlobalVectorType valueInTesla( const GlobalPointType& p) const = 0;

  /** Returns the maximal order of available derivatives.
   *  Returns 0 if derivatives are not available.
   */
  virtual int hasDerivatives() const {return false;}

  /** Returns the Nth spacial derivative of the field in the local frame.
   */
  virtual LocalVectorType derivativeInTeslaPerMeter( const LocalPointType& p, 
						     int N) const {
    return LocalVectorType();
  }

};

#endif
