//#include "Utilities/Configuration/interface/Architecture.h"
#include "CalibTracker/SiStripConnectivity/interface/ApvPairConnection.h"
using namespace std;

ApvPairConnection::ApvPairConnection(ApvConnection apv1_, 
				     ApvConnection apv2_){
  apv1 = apv1_;
  apv2 = apv2_;

  consistencyChecks();

  fedNumber = -1;
  fedChannel = -1;
  

}

bool ApvPairConnection::consistencyChecks(){
  //
  // consistency checks
  //
  
  if (apv1.getFecSlot() != apv2.getFecSlot()){  
    return false;
  }
  else
    fecSlot = apv1.getFecSlot();
  
  if (apv1.getRingSlot() != apv2.getRingSlot()){
    return false;
  }else
    ringSlot = apv1.getRingSlot();
  
  if (apv1.getCcuAddress() != apv2.getCcuAddress()){
    return false;
  }else
    ccuAddress = apv1.getCcuAddress();
  
  if (apv1.getI2CChannel() != apv2.getI2CChannel()){
    return false;
  }else
    i2cChannel = apv1.getI2CChannel();
  
  i2cAddressApv1 = apv1.getI2CAddress();
  i2cAddressApv2 = apv2.getI2CAddress();
  
  if (i2cAddressApv1 > i2cAddressApv2){
    int temp;
    temp = i2cAddressApv1;
    i2cAddressApv1 = i2cAddressApv2;
    i2cAddressApv2 = temp;
    
    ApvConnection temp2 = apv1;
    apv1=apv2;
    apv2 = temp2;
  }
  
  if (i2cAddressApv1 != 32 && 
      i2cAddressApv1 != 34 && 
      i2cAddressApv1 != 36 ){
    //    cout <<" Error - Apv1 must be in channel 32, 34 or 36"<<endl;
    return false;
  }
  if (i2cAddressApv2 != 33 && 
      i2cAddressApv2 != 35 && 
      i2cAddressApv2 != 37 ){
    //    cout <<" Error - Apv2 must be in channel 33, 35 or 37"<<endl;
    return false;
  }
  
  if((i2cAddressApv1 == 32 &&i2cAddressApv2 != 33) ||
     (i2cAddressApv1 == 34 &&i2cAddressApv2 != 35) ||
     (i2cAddressApv1 == 36 &&i2cAddressApv2 != 37) ) {
    //    cout <<" Error - Apv Pair badly formed: Apv1 "<<i2cAddressApv1<<" Apv2 "<<i2cAddressApv2<<endl;
    return false;
  }
  return true;
}
