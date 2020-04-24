#ifndef CSC_GANGED_WIRE_GROUPING_H
#define CSC_GANGED_WIRE_GROUPING_H

/** \class CSCGangedWireGrouping
 * A concrete CSCWireGrouping in which wires are ganged.
 *
 * All 'wire'-related values refer to 'virtual wires' which cover the
 * face of the detector. All 'wire-group' values refer to actual
 * active (in principle) read out channels.
 *
 * \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/src/CSCWireGrouping.h"
#include <vector>

class CSCGangedWireGrouping : public CSCWireGrouping {

 public:

  typedef std::vector<int> Container;
  typedef Container::const_iterator CIterator;

  ~CSCGangedWireGrouping() override {}
     
  /**
   * Constructor from endcap muon wire information parsed from DDD
   */
  CSCGangedWireGrouping( const Container& consecutiveGroups, 
		 const Container& wiresInConsecutiveGroups, 
		 int numberOfGroups );

  /**
   * Total number of (virtual) wires.
   * Some wires may not be implemented in the hardware.
   * This is the number which would fill the region covered
   * by wires, assuming the constant wire spacing.
   */
  int numberOfWires() const override {
    return theNumberOfWires; }

  /**
   * How many wire groups
   */
  int numberOfWireGroups() const override {
    return theNumberOfGroups; }

  /**
   * How many wires in a wiregroup
   */
  int numberOfWiresPerGroup( int wireGroup ) const override;

  /**
   * Wire group containing a given wire
   */
  int wireGroup( int wire ) const override;

  /**
   * Middle of wire-group.
   * This is the central wire no. for a group with an odd no. of wires.
   * This is a pseudo-wire no. for a group with an even no. of wires.
   * Accordingly, it is non-integer.
   */
  float middleWireOfGroup( int wireGroup ) const override;

  /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   */
  CSCWireGrouping* clone() const override {
    return new CSCGangedWireGrouping(*this);
  }

 private:
  // Expanded information from DDD
  int theNumberOfWires;
  int theNumberOfGroups;
  Container theFirstWireOfEachWireGroup;
  Container theNumberOfWiresPerWireGroup;

};

#endif
