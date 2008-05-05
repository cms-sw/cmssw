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

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

private:

  AlignmentPositionError* theAlignmentPositionError;

};

#endif // ALIGNABLE_DET_H
