#ifndef VolumeSide_H
#define VolumeSide_H

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

/** Class for delimiding surface of a volume.
 *  The additional information with respect to Surface that is needed
 *  to define the volume is <BR>
 *  a) which side of the Surface the volume is (enumerator Surface::Side) <BR>
 *  b) which face of the volume this surface represents (enumerator GlobalFace). 
 *     Only 6 possible values for volume face are defined.
 */

class VolumeSide {
public:
  typedef SurfaceOrientation::GlobalFace GlobalFace;
  typedef SurfaceOrientation::Side Side;
  
  typedef ReferenceCountingPointer<Surface>    SurfacePointer;

  VolumeSide( Surface* surf, GlobalFace gSide, Side sSide) : 
    theSurface( surf),  theGlobalFace( gSide), theSurfaceSide( sSide) {}

  VolumeSide( SurfacePointer surf, GlobalFace gSide,
	      Side sSide) : 
    theSurface( surf),  theGlobalFace( gSide), theSurfaceSide( sSide) {}

  Surface& mutableSurface() const {return *theSurface;}

  const Surface& surface() const {return *theSurface;}

  GlobalFace globalFace() const { return theGlobalFace;}

  Side  surfaceSide() const {return theSurfaceSide;}

private:

  SurfacePointer theSurface;
  GlobalFace     theGlobalFace;
  Side           theSurfaceSide;

};

#endif
