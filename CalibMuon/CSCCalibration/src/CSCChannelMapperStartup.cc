#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperStartup.h"

int CSCChannelMapperStartup::rawStripChannel( const CSCDetId& id, int igeo ) const {

  // Translate a geometry-oriented strip channel in range 1-80, igeo,
  // into corresponding raw channel.

  int iraw = igeo;

  bool zplus = (id.endcap()==1);

  bool me1a = (id.station()==1) && (id.ring()==4);
  bool me1b = (id.station()==1) && (id.ring()==1);

  if ( me1a && zplus ) { iraw = 17 - iraw; } // 1-16 -> 16-1
  if ( me1b && !zplus) { iraw = 65 - iraw; }  // 1-64 -> 64-1
  if ( me1a ) { iraw += 64 ;} // set 1-16 to 65-80

  return iraw;
}


int CSCChannelMapperStartup::geomStripChannel( const CSCDetId& id, int iraw ) const {
  // Translate a raw strip channel in range 1-80, iraw,  into
  // corresponding geometry-oriented channel in which increasing
  // channel number <-> strip number increasing with +ve local x.

  int igeo = iraw;

  bool zplus = (id.endcap()==1);
  bool me11 = (id.station()==1) && (id.ring()==1);
  bool me1a = me11 && (iraw > 64);
  bool me1b = me11 && (iraw <= 64);

  if ( me1a ) igeo -= 64; // 65-80 -> 1-16
  //if ( me1a ) igeo %= 64; // 65-80 -> 1-16
  if ( me1a && zplus ) { igeo = 17 - igeo; } // 65-80 -> 16-1
  if ( me1b && !zplus) { igeo = 65 - igeo; }  // 1-64 -> 64-1

  return igeo;
}

int CSCChannelMapperStartup::channelFromStrip( const CSCDetId& id, int strip ) const {
  // This just returns the electronics channel label to which a given strip is connected
  // In all chambers but ME1A this is just a direct 1-1 correspondence.
  // In ME1A the 48 strips are ganged into 16 channels: 1+17+33->1, 2+18+34->2, ... 16+32+48->16.
  int ichan = strip;
  bool me1a = (id.station()==1) && (id.ring()==4);
  if ( me1a && strip>16 ) ichan = (strip-1)%16 + 1; // gang the 48 to 16
  return ichan;
}

CSCDetId CSCChannelMapperStartup::rawCSCDetId( const CSCDetId& id ) const {
  // Return the effective online CSCDetId for given offline CSCDetId
  // That means the same one except for ME1A, which online is part of ME11 (channels 65-80)
  CSCDetId idraw( id );
  bool me1a = (id.station()==1) && (id.ring()==4);
  if ( me1a ) idraw = CSCDetId( id.endcap(), id.station(), 1, id.chamber(), id.layer() );
  return idraw;
}
