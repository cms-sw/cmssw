#ifndef Alignment_CommonAlignment_AlignableDet_h
#define Alignment_CommonAlignment_AlignableDet_h

#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

/// An AlignableComposite that has AlignableDetUnits as direct component.

class AlignableDet: public AlignableComposite 
{

public:
  
  /// Constructor (copies  GeomDetUnits of GeomDet)
  AlignableDet( const GeomDet* geomDet );
  
  /// Destructor
  ~AlignableDet();
  
  /// Return vector of components
  virtual std::vector<Alignable*> components() const ;

  /// Return given AlignableDetUnit
  AlignableDetUnit &detUnit(int i);

  /// Set alignment position error of this and all components to given error
  virtual void setAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  virtual void addAlignmentPositionError(const AlignmentPositionError& ape);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame
  virtual void addAlignmentPositionErrorFromRotation(const RotationType& rot);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the local reference frame
  virtual void addAlignmentPositionErrorFromLocalRotation(const RotationType& rot);


  /// Alignable object identifier
  virtual int alignableObjectId () const { return AlignableObjectId::AlignableDet; }

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

private:

  /// Container of components
  std::vector<AlignableDetUnit*> theDetUnits ;

  float theWidth, theLength;

  AlignmentPositionError* theAlignmentPositionError;

};




#endif // ALIGNABLE_DET_H

