#ifndef CSC_WIRE_GROUPING_H
#define CSC_WIRE_GROUPING_H

/** \class CSCWireGrouping
 * An ABC defining interface for wire-grouping related functionality
 * for detectors modelled by a WireTopology.
 *
 * \author Tim Cox
 *
 */

class CSCWireGrouping {
public:
  virtual ~CSCWireGrouping() {}

  /**
   * Total number of (virtual) wires.
   * Some wires may not be implemented in the hardware.
   * This is the number which would fill the region covered
   * by wires, assuming the constant wire spacing.
   */
  virtual int numberOfWires() const = 0;

  /**
   * How many wire groups
   */
  virtual int numberOfWireGroups() const = 0;

  /**
   * How many wires in a wiregroup
   */
  virtual int numberOfWiresPerGroup(int wireGroup) const = 0;

  /**
   * Wire group containing a given wire
   */
  virtual int wireGroup(int wire) const = 0;

  /**
   * Middle of wire-group.
   * This is the central wire no. for a group with an odd no. of wires.
   * This is a pseudo-wire no. for a group with an even no. of wires.
   * Accordingly, it is non-integer.
   */
  virtual float middleWireOfGroup(int wireGroup) const = 0;

  /**
   * Allow proper copying of derived classes via base pointer
   */
  virtual CSCWireGrouping* clone() const = 0;
};

#endif
