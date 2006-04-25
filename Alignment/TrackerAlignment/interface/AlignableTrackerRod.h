#ifndef Alignment_TrackerAlignment_AlignableTrackerRod_H
#define Alignment_TrackerAlignment_AlignableTrackerRod_H


#include <iosfwd> 
#include <iostream>
#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A composite of AlignableDets forming a 1-dimensional rod


class AlignableTrackerRod: public AlignableComposite 
{

 public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerRod & ); 
  

  /// Constructor from geomdets of the rod's components
  AlignableTrackerRod( std::vector<GeomDet*>& geomDets );
  
  ~AlignableTrackerRod();
  
  virtual std::vector<Alignable*> components() const ;

  AlignableDet &det(int i);

  virtual float length() const;

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableRod;
  }

 private:
  // gets the global position as the average over all Dets in the Rod
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets;


};


#endif  // ALIGNABLE_ROD_H


