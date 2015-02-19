#ifndef Alignment_CommonAlignment_AlignableDet_h
#define Alignment_CommonAlignment_AlignableDet_h

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// An AlignableComposite corresponding to a composite GeomDet
/// direct components are AlignableDetUnits or AlignableDets.
class AlignableDet: public AlignableComposite 
{

public:
  
  /// Constructor: If addComponents = true, creates components for
  /// geomDet's components, assuming they are GeomDetUnits
  AlignableDet( const GeomDet* geomDet, bool addComponents = true );
  
  /// Destructor
  virtual ~AlignableDet();

  /// Set the AlignmentPositionError and, if (propagateDown), to all components
  virtual void setAlignmentPositionError(const AlignmentPositionError &ape, bool propagateDown);

  /// Add (or set if it does not exist yet) the AlignmentPositionError,
  /// if (propagateDown), add also to all components
  virtual void addAlignmentPositionError(const AlignmentPositionError &ape, bool propagateDown);

  /// Add (or set if it does not exist yet) the AlignmentPositionError
  /// resulting from a rotation in the global reference frame,
  /// if (propagateDown), add also to all components
  virtual void addAlignmentPositionErrorFromRotation(const RotationType &rot, bool propagateDown);

  // No need to overwrite, version from AlignableComposite is just fine:
  // virtual void addAlignmentPositionErrorFromLocalRotation(const RotationType &rot,
  //							  bool propagateDown);

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrorsExtended* alignmentErrors() const;

  /// alignment position error - for checking only, otherwise use alignmentErrors() above!  
  const AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError;}

private:

  AlignmentPositionError* theAlignmentPositionError;

};

#endif // ALIGNABLE_DET_H
