#ifndef CSCChannelMapperStartup_H
#define CSCChannelMapperStartup_H

/**
 * \class CSCChannelMapperStartup
 *
 * A concrete CSCChannelMapper class to
 * map between raw/online channel numbers (for strips/cathodes and wires/anodes)
 * and offline geometry-oriented channel numbers, in which increasing number
 * corresponds to increasing local x (strips) or y (wire groups) as defined
 * in CMS Note CMS IN-2007/024.
 *
 * This version is for CMS Startup (2008-2013)
 *
 * 1. Sorts out readout-flipping within the two endcaps for ME1a and ME1b strip channels. <BR>
 * 2. Maps the ME1a channels from online labels 65-80 to offline 1-16. <BR>
 * 3. Does nothing with wiregroup channels; the output = the input. <BR>
 *
 * Since ME1a is ganged, the 48 strips in ME1a are fed to 16 channels, so it is important
 * to distinguish the nomenclatures "strip" vs "channel". It is usually a meaningful distinction!
 *
 * Also note that the CSCDetId for ME11 and ME1b is identical. Offline we presume ring=1 of station 1
 * to mean the ME1b strips. We use the identifier ring=4 to denote the ME1a strips.
 *
 * \author Tim Cox
 *
 */

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"

class CSCChannelMapperStartup : public CSCChannelMapperBase {
 public:

  CSCChannelMapperStartup() {}
  ~CSCChannelMapperStartup() {}

  virtual std::string name() const {return "CSCChannelMapperStartup";}

  /// Return raw strip channel number for input geometrical channel number
  int rawStripChannel( const CSCDetId& id, int igeom ) const;

  /// Return geometrical strip channel number for input raw channel number
  int geomStripChannel( const CSCDetId& id, int iraw ) const ;

  /// Offline conversion of a strip (geometric labelling) back to channel
  /// (Startup: convert the 48 strips of ME1A to 16 ganged channels.)
  int channelFromStrip( const CSCDetId& id, int strip ) const;

  /// Construct raw CSCDetId matching supplied offline CSCDetid
  /// (Startup:  return the ME11 CSCDetID when supplied with that for ME1A)
  CSCDetId rawCSCDetId( const CSCDetId& id ) const;

};

#endif
