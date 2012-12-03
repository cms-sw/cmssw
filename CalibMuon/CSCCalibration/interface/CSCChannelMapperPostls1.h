#ifndef CSCChannelMapperPostls1_H
#define CSCChannelMapperPostls1_H

/**
 * \class CSCChannelMapperPostls1
 *
 * A concrete CSCChannelMapper class to
 * map between raw/online channel numbers (for strips/cathodes and wires/anodes)
 * and offline geometry-oriented channel numbers, in which increasing number
 * corresponds to increasing local x (strips) or y (wire groups) as defined
 * in CMS Note CMS IN-2007/024.
 *
 * This version is for CMS Postls1 (2013-)
 *
 * 1. Sorts out readout-flipping within the two endcaps for ME1a and ME1b strip channels. <BR>
 *    We do not yet know whether there WILL be any flipping. For now we presume it is as in the Startup case. <BR>
 * 2. Doesnothing with ME1a channels since we intend each of the 48 strips to go to 48 individual channels. <BR>
 * 3. Does nothing with wiregroup channels; the output = the input. <BR>
 *
 * Also note that the CSCDetId for ME11 and ME1b is identical. Offline we presume ring=1 of station 1
 * to mean the ME1b strips. We use the identifier ring=4 to denote the ME1a strips.
 *
 * \author Tim Cox
 *
 */

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"

class CSCChannelMapperPostls1 : public CSCChannelMapperBase {
 public:

  CSCChannelMapperPostls1() {}
  ~CSCChannelMapperPostls1() {}

  virtual std::string name() const {return "CSCChannelMapperPostls1";}

  /// Return raw strip channel number for input geometrical channel number
  int rawStripChannel( const CSCDetId& id, int igeom ) const;

  /// Return geometrical strip channel number for input raw channel number
  int geomStripChannel( const CSCDetId& id, int iraw ) const ;

  /// Offline conversion of a strip (geometric labelling) back to channel
  /// (Postls1: 1-1 correspondence strip to channel)
  int channelFromStrip( const CSCDetId& id, int strip ) const;

  /// Construct raw CSCDetId matching supplied offline CSCDetid
  /// (Postls1: leave ME1a detid alone)
  CSCDetId rawCSCDetId( const CSCDetId& id ) const;

};

#endif
