#ifndef RecoTauTag_TauTagTools_ECALBounds_H
#define RecoTauTag_TauTagTools_ECALBounds_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

class BoundCylinder;
class BoundDisk;
class Cylinder;
class Disk;

class ECALBounds {
public:
  static const Surface& barrelBound()    {check(); return *theCylinder;}
  static const Surface& negativeEndcapDisk() {check(); return *theNegativeDisk;}
  static const Surface& positiveEndcapDisk() {check(); return *thePositiveDisk;}
  /** Hard-wired numbers defining the envelope of the sensitive volumes.
   */
  static float barrel_innerradius()     {return 129.0f;}
  static float barrel_outerradius()     {return 175.f;}
  static float barrel_halfLength() {return 270.89f;}
  static float endcap_innerradius()     {return 31.6f;}
  static float endcap_outerradius()     {return 171.1f;}
  static float endcap_innerZ() {return 314.40f;}
  static float endcap_outerZ() {return 388.f;}    
 private:
  static ReferenceCountingPointer<Surface> theCylinder;
  static ReferenceCountingPointer<Surface> theNegativeDisk;
  static ReferenceCountingPointer<Surface> thePositiveDisk;
  static bool theInit;
  static void check() {if (!theInit) initialize();}
  static void initialize();
};

#endif
