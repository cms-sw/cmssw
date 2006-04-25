#ifndef Alignment_CommonAlignment_AlignableDet_h
#define Alignment_CommonAlignment_AlignableDet_h

#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

/// An AlignableComposite that has AlignableDetUnits as direct component.

class AlignableDet: public AlignableComposite 
{
 public:

  AlignableDet( GeomDet* geomDet );

  ~AlignableDet();
  
  virtual std::vector<Alignable*> components() const ;

  AlignableDetUnit &geomDetUnit(int i);
 
  virtual void setAlignmentPositionError(const AlignmentPositionError& ape);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableDet;
  }

  /// Movement with respect to the GLOBAL CMS reference frame. 
  /// The corresponding Det is not moved (done via the components = DetUnits) 
  virtual void move( const GlobalVector& displacement) 
  {
    moveAlignableOnly(displacement);
  }

  /// Rotation with respect to the GLOBAL CMS reference frame. 
  /// The corresponding Det is not rotated
  /// (done via the components = DetUnits)
  virtual void rotateInGlobalFrame( const RotationType& rotation) 
  {
    rotateAlignableOnly(rotation);
  }

private:
  std::vector<AlignableDetUnit*> theDetUnits ;

};




#endif // ALIGNABLE_DET_H

