#ifndef CSCChannelMapperBase_H
#define CSCChannelMapperBase_H

/**
 * \class CSCChannelMapperBase
 *
 * Base class for concrete CSCChannelMapper classes that
 * map between raw/online channel numbers (for strips/cathodes and wires/anodes)
 * and offline geometry-oriented channel numbers, in which increasing number
 * corresponds to increasing local x (strips) or y (wire groups) as defined
 * in CMS Note CMS IN-2007/024.
 *
 * The original of this class, CSCChannelTranslator, was written in the expectation
 * that one day it would be replaced by a full "cable map" stored in conditions data.
 * That has not yet been required and so the mappings are hard-coded.
 *
 * Concrete derived classes must implement the following: <BR>
 * 1. Sort out any readout-flipping within the two endcaps for ME1a and ME1b strip channels. <BR>
 * 2. If ME1a is ganged then map the ME1a channels from online labels 65-80 to offline 1-16. <BR>
 * 3. Do nothing with wiregroup channels; the output = the input. <BR>
 * (Historically some test beam data needed wiregroup remapping but this was embedded directly in the
 * Unpacker of CSCRawToDigi. We want to move any such mappings into this class rather than have them 
 * scattered through the code.)
 *
 * Beware that if ME1a is ganged,the 48 strips in ME1a are fed to 16 channels, so it is important
 * to distinguish the nomenclatures "strip" vs "channel". It is usually a meaningful distinction!
 *
 * Also note that the CSCDetId for ME11 and ME1b is identical. Offline we presume ring=1 of station 1
 * to mean the ME1b strips. We use the identifier ring=4 to denote the ME1a strips.
 *
 * \author Tim Cox
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCChannelMapperBase{
 public:

  CSCChannelMapperBase() {}
  virtual ~CSCChannelMapperBase() {}

  virtual std::string name() const {return "CSCChannelMapperBase";}

  /// Return raw strip channel number for input geometrical channel number
  virtual int rawStripChannel( const CSCDetId& id, int igeom ) const = 0;
  /// Return raw wiregroup channel number for input geometrical channel number
  int rawWireChannel( const CSCDetId& id, int igeom ) const { return igeom; }
  /// Return geometrical strip channel number for input raw channel number
  virtual int geomStripChannel( const CSCDetId& id, int iraw ) const = 0;
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

  /// Offline conversion of a strip (geometric labelling) back to channel
  /// (At present this just has to convert the 48 strips of ME1A to 16 ganged channels.)
  virtual int channelFromStrip( const CSCDetId& id, int strip ) const = 0;

  /// Construct raw CSCDetId matching supplied offline CSCDetid
  /// (At present all this has to do is return the ME11 CSCDetID when supplied with that for ME1A)
  virtual CSCDetId rawCSCDetId( const CSCDetId& id ) const = 0;

};

#endif
