#ifndef Alignment_CommonAlignment_AlignableExtras_H
#define Alignment_CommonAlignment_AlignableExtras_H

/** \class AlignableExtras
 *
 * A container for additional/extra alignables
 *
 *  Original author: Andreas Mussgiller, August 2010
 *
 *  $Date: 2010/09/10 10:26:20 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"

class AlignableExtras 
{
 public:
  
  typedef align::Alignables Alignables;

  /// Constructor (builds all alignables)
  explicit AlignableExtras();

  /// Return alignables determined by name
  Alignables& subStructures(const std::string &subStructName) {
    return alignableLists_.find(subStructName);
  }

  /// Return beam spot alignable as a vector with one element
  Alignables& beamSpot() { return this->subStructures("BeamSpot");}

  Alignables components() const { return components_; }

  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrors* alignmentErrors() const;

  void dump(void) const;

  /// Initialize the alignable beam spot with the given parameters
  void initializeBeamSpot(double x, double y, double z,
			  double dxdz, double dydz);

 private:
  
  AlignSetup<Alignables> alignableLists_; //< kind of map of lists of alignables
  Alignables components_; //< list of alignables
};

#endif //AlignableExtras_H
