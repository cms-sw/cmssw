#include "DQM/SiStripCommon/interface/SiStripControlDirPath.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
//
string SiStripControlDirPath::path( // unsigned short crate,
				   unsigned short slot,
				   unsigned short ring,
				   unsigned short addr,
				   unsigned short chan
				   // unsigned short i2c 
				   ) { 
  stringstream folder;
  folder << control();
  //   if ( crate != all_ ) {
  //     folder << sep_ << crate_ << crate;
  if ( slot != all_ ) {
    folder << sep_ << slot_ << slot;
    if ( ring != all_ ) {
      folder << sep_ << ring_ << ring;
      if ( addr != all_ ) {
	folder << sep_ << addr_ << addr;
	if ( chan != all_ ) {
	  folder << sep_ << chan_ << chan;
	  // 	    if ( i2c != all_ ) {
	  // 	      folder << sep_ << i2c_ << i2c;
	  // 	    }
	}
      }
    }
  }
  //   }
  return folder.str();
}


