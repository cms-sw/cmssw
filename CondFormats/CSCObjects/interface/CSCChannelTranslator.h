#ifndef CSCChannelTranslator_h
#define CSCChannelTranslator_h

/**
 * \class CSCChannelTranslator
 *
 * Maps between raw/online channel numbers (for strips/cathodes and wires/anodes)
 * and offline geometry-oriented channel numbers, in which increasing number
 * corresponds to increasing local x (strips) or y (wire groups) as defined
 * in CMS Note CMS IN-2007/024.
 *
 * It is expected that this class will one day need to make use of a 'cable map' stored
 * in conditions data. At present it does not, and the mappings are hard-coded.
 *
 * Currently this class does the following: <BR>
 * 1. Sorts out the readout-flipping within the two endcaps for ME1a and ME1b strip channels. <BR>
 * 2. Maps the ME1a channels from online labels 65-80 to offline 1-16. It is expected that in the long
 * run we may drop the offline convention for ME1a as an independent CSCDetId but this has not yet
 * been done. When it is, this conversion will be removed. <BR>
 * 3. Does nothing with wiregroup channels; the output = the input. In the long term we intend to
 * move remapping currently embedded in the CSCRawToDigi unpacker into this class.<BR>
 *
 * \author Tim Cox
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCChannelTranslator{
 public:
  CSCChannelTranslator() {}
  ~CSCChannelTranslator() {}

  /// Return raw strip channel number for input geometrical channel number
  int rawStripChannel( const CSCDetId& id, int igeom ) const;
  /// Return raw wiregroup channel number for input geometrical channel number
  int rawWireChannel( const CSCDetId& id, int igeom ) const { return igeom; }
  /// Return geometrical strip channel number for input raw channel number
  int geomStripChannel( const CSCDetId& id, int iraw ) const ;
  /// Return geometrical wiregroup channel number for input raw channel number
  int geomWireChannel( const CSCDetId& id, int iraw ) const { return iraw; }

  /// Alias for rawStripChannel
  int rawCathodeChannel( const CSCDetId& id, int igeom ) const { return rawStripChannel( id, igeom );}
  /// Alias for rawWireChannel
  int rawAnodeChannel( const CSCDetId& id, int igeom ) const { return rawWireChannel( id, igeom );}
  /// Alias for geomStripChannel
  int geomCathodeChannel( const CSCDetId& id, int iraw ) const { return geomStripChannel( id, iraw );}
  /// Alias for geomWireChannel
  int geomAnodeChannel( const CSCDetId& id, int iraw ) const { return geomWireChannel( id, iraw );}

};

#endif
