#ifndef CSC_UNGANGED_WIRE_GROUPING_H
#define CSC_UNGANGED_WIRE_GROUPING_H

/** \class CSCUngangedWireGrouping
 * A concrete CSCWireGrouping in which wires are not ganged.
 *
 * \author Tim Cox
 *
 */

#include "Geometry/CSCGeometry/src/CSCWireGrouping.h"

class CSCUngangedWireGrouping : public CSCWireGrouping {
 public:
  virtual ~CSCUngangedWireGrouping() {}
  explicit CSCUngangedWireGrouping( int nwires ) : 
      theNumberOfWires( nwires ) {}

  /**
   * Total number of (virtual) wires.
   * Some wires may not be implemented in the hardware.
   * This is the number which would fill the region covered
   * by wires, assuming the constant wire spacing.
   */
  int numberOfWires() const {
    return theNumberOfWires; }

  /**
   * How many wire groups. Unganged so #groups = #wires.
   */
  int numberOfWireGroups() const {
    return numberOfWires(); }

  /**
   * How many wires in a wiregroup. Unganged so 1 wire/group.
   */
  int numberOfWiresPerGroup( int wireGroup ) const {
    return 1; }

  /**
   * Wire group containing a given wire. Unganged means wire group is wire.
   */
  int wireGroup(int wire) const {
    return wire; }

  /**
   * Middle of wire-group.
   * This is the central wire no. for a group with an odd no. of wires.
   * This is a pseudo-wire no. for a group with an even no. of wires.
   * Accordingly, it is non-integer.
   * Unganged, wire group is wire is middle!
   */
  float middleWireOfGroup( int wireGroup ) const {
    return static_cast<float>( wireGroup ); }

 /**
   * Clone to handle correct copy of component objects referenced
   * by base class pointer.
   */
  CSCWireGrouping* clone() const {
    return new CSCUngangedWireGrouping(*this);
  }

 private:
  int theNumberOfWires;

};

#endif
