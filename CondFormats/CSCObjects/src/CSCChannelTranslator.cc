#include "CondFormats/CSCObjects/interface/CSCChannelTranslator.h"

int CSCChannelTranslator::rawStripChannel( const CSCDetId& id, int igeo ) const {

  // Translate a geometry-oriented strip channel in range 1-80, igeo,
  // into corresponding raw channel.

  int iraw = igeo;

  bool zplus = (id.endcap()==1); 

  // While we have separate detids for ME1a and ME1b...
  bool me1a = (id.station()==1) && (id.ring()==4);
  bool me1b = (id.station()==1) && (id.ring()==1);

  if ( me1a && zplus ) { iraw = 17 - iraw; } // 1-16 -> 16-1
  if ( me1b && !zplus) { iraw = 65 - iraw; }  // 1-64 -> 64-1
  if ( me1a ) { iraw += 64 ;} // set 1-16 to 65-80 

  return iraw;
}


int CSCChannelTranslator::geomStripChannel( const CSCDetId& id, int iraw ) const {
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

