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
  
  /// Constructor (copies  GeomDetUnits of GeomDet)
  AlignableDet( GeomDet* geomDet );
  
  /// Destructor
  ~AlignableDet();
  
  /// Return vector of components
  virtual std::vector<Alignable*> components() const ;

  /// Return given geomDetUnit
  AlignableDetUnit &geomDetUnit(int i);
 
  /// Set alignment position error of all components to given error
  virtual void setAlignmentPositionError(const AlignmentPositionError& ape);

  /// Alignable object identifier
  virtual int alignableObjectId () const { return AlignableObjectId::AlignableDet; }

  /// Return vector of alignment data
  virtual Alignments* alignments() const;

  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const;

private:

  /// Container of components
  std::vector<AlignableDetUnit*> theDetUnits ;

};




#endif // ALIGNABLE_DET_H

