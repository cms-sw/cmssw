#include "EventFilter/ESRawToDigi/interface/ESElectronicsMapper.h"

ESElectronicsMapper::ESElectronicsMapper(const ParameterSet& ps) {

  lookup_ = ps.getUntrackedParameter<FileInPath>("LookupTable");
 
  // read in look-up table
  int nLines, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {

    file >> nLines;

    for (int i=0; i<nLines; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      
      if (iz==-1) z = 2;
      else z = iz;
      
      fed_[z-1][ip-1][ix-1][iy-1] = fed;
    }

  } else {
    cout<<"ESElectronicsMapper::ESElectronicsMapper : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<endl;
  }

}

int ESElectronicsMapper::getFED(int zside, int plane, int x, int y) { 

  return fed_[zside-1][plane-1][x-1][y-1]; 

} 
