#ifndef RecoTauTag_TauTagTools_ECALBounds_H
#define RecoTauTag_TauTagTools_ECALBounds_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
 
/** A definition of the ECAL inner surface.
  *  The information is not automatically computed from the
  *  geometry, but is hard-coded in this class.
  * 
  *  Ported from ORCA
  */

class ECALBounds {
public:

  static const BoundCylinder& barrelBound()    { return *theCylinder; }
  static const BoundDisk& negativeEndcapDisk() { return *theNegativeDisk; }
  static const BoundDisk& positiveEndcapDisk() { return *thePositiveDisk; }

  /** Hard-wired numbers defining the envelope of the sensitive volumes.
   */
  static float barrel_innerradius() { return 129.0f; }
  static float barrel_outerradius() { return 175.f; }
  static float barrel_halfLength()  { return 270.89f; }
  static float endcap_innerradius() { return 31.6f; }
  static float endcap_outerradius() { return 171.1f; }
  static float endcap_innerZ()      { return 314.40f; }
  static float endcap_outerZ()      { return 388.f; }

  /** Hard-wired numbers defining eta cracks.
   */
  static std::pair<float,float> crack_absEtaIntervalA() { return std::pair<float,float>(0.000f, 0.018f); }
  static std::pair<float,float> crack_absEtaIntervalB() { return std::pair<float,float>(0.423f, 0.461f); }
  static std::pair<float,float> crack_absEtaIntervalC() { return std::pair<float,float>(0.770f, 0.806f); }
  static std::pair<float,float> crack_absEtaIntervalD() { return std::pair<float,float>(1.127f, 1.163f); }
  static std::pair<float,float> crack_absEtaIntervalE() { return std::pair<float,float>(1.460f, 1.558f); }

 private:
  
  static const ReferenceCountingPointer<BoundCylinder> theCylinder;
  static const ReferenceCountingPointer<BoundDisk>     theNegativeDisk;
  static const ReferenceCountingPointer<BoundDisk>     thePositiveDisk;
};

#endif
