#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperPostls1.h"

int CSCChannelMapperPostls1::rawStripChannel( const CSCDetId& id, int igeo ) const {

  // Translate a geometry-oriented strip channel in range 1-80, igeo,
  // into corresponding raw channel.

  int iraw = igeo;

  bool zplus = (id.endcap()==1);

  bool me1a = (id.station()==1) && (id.ring()==4);
  bool me1b = (id.station()==1) && (id.ring()==1);

  if ( me1a && zplus ) { iraw = 49 - iraw; } // 1-48 -> 48-1
  if ( me1b && !zplus) { iraw = 65 - iraw; }  // 1-64 -> 64-1

  return iraw;
}


int CSCChannelMapperPostls1::geomStripChannel( const CSCDetId& id, int iraw ) const {
  // Translate a raw strip channel in range 1-80, iraw,  into
  // corresponding geometry-oriented channel in which increasing
  // channel number <-> strip number increasing with +ve local x.

  int igeo = iraw;

  bool zplus = (id.endcap()==1);
  bool me1a = (id.station()==1) && (id.ring()==4);
  bool me1b = (id.station()==1) && (id.ring()==1);

  if ( me1a && zplus ) { igeo = 49 - igeo; } // 1-48 -> 48-1
  if ( me1b && !zplus) { igeo = 65 - igeo; }  // 1-64 -> 64-1

  return igeo;
}

int CSCChannelMapperPostls1::channelFromStrip( const CSCDetId& id, int strip ) const {
  // This just returns the electronics channel label to which a given strip is connected
  // In all chambers (including upgraded ME1A) this is just a direct 1-1 correspondence.
  int ichan = strip;
  return ichan;
}

CSCDetId CSCChannelMapperPostls1::rawCSCDetId( const CSCDetId& id ) const {
  // Return the effective online CSCDetId for given offline CSCDetId
  // That means the same one (for upgraded ME1A)
  CSCDetId idraw( id );
  return idraw;
}
