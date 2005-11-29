//#include "Utilities/Configuration/interface/Architecture.h"
#include "CalibTracker/SiStripConnectivity/interface/ApvConnection.h"
//#include "Utilities/Notification/interface/Verbose.h"
#include <iostream>
ApvConnection::ApvConnection(){
  //
  // set these to -1, meaning no settings are done
  //
  fecSlot =-1;
  ringSlot =-1;
  ccuAddress =-1;
  i2cChannel =-1;
  i2cAddress =-1;
}

ApvConnection::ApvConnection(
			     int fs, int rs,
			     int ca, int ic, int ia){
  fecSlot= fs;
  ringSlot =rs;
  ccuAddress =ca;
  i2cChannel =ic;
  i2cAddress =ia;
  //  if (debugV) cout << "Creating APV with ccuAddress " << ca << " i2cChannel " << ic
  //                   << " i2cAddress " << ia << endl;
}
bool ApvConnection::operator==(const ApvConnection& apv){   
  return (getFecSlot()    == apv.getFecSlot() &&
          getRingSlot()   == apv.getRingSlot() &&
          getCcuAddress() == apv.getCcuAddress() &&
          getI2CChannel() == apv.getI2CChannel() &&
          getI2CAddress() == apv.getI2CAddress());
}

