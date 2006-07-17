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
  AlignableTrackerRod( std::vector<const GeomDet*>& geomDets );
  
  /// Destructor
  ~AlignableTrackerRod();
  
  /// Return all components of the rod
  virtual std::vector<Alignable*> components() const ;

  /// Return AlignableDet at given index
  AlignableDet &det(int i);

  /// Return length calculated from components
  virtual float length() const;

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableRod; }

 private:
  // Return the global position as the average over all Dets in the Rod
  PositionType computePosition(); 
  // Return the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // Return the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets;


};


#endif  // ALIGNABLE_ROD_H


