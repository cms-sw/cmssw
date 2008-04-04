#ifndef Alignment_MuonAlignment_AlignableDTChamber_H
#define Alignment_MuonAlignment_AlignableDTChamber_H

/** \class AlignableDTChamber
 *  The alignable muon DT chamber.
 *
 *  $Date: 2007/12/06 01:30:57 $
 *  $Revision: 1.10.4.1 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include <iosfwd> 
#include <iostream>
#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"


/// A muon DT Chamber (has an associated GeomDet)


class AlignableDTChamber: public AlignableComposite
{

 public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  friend std::ostream& operator << ( std::ostream &, const AlignableDTChamber & ); 
  

  /// Constructor from corresponding geomdet
  AlignableDTChamber( const GeomDet* geomDet  );
  
  /// Destructor
  ~AlignableDTChamber();
  
  /// Return all direct components (superlayers)
  virtual std::vector<Alignable*> components() const ;

  /// Return component (superlayer) at given index
  AlignableDet &det(int i);

  /// Return length of alignable
  virtual float length() const { return theLength; }

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
  virtual StructureType alignableObjectId () const { return align::AlignableDTChamber; }

  //virtual void twist(float);

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

 private:

  std::vector<AlignableDet*> theDets;

  float theWidth, theLength;

  AlignmentPositionError* theAlignmentPositionError;

};


#endif  // ALIGNABLE_DT_CHAMBER_H


