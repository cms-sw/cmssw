#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_APVPAIRCONNECTION_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_APVPAIRCONNECTION_H

#include "CalibTracker/SiStripConnectivity/interface/ApvConnection.h"
#include <iostream>
class ApvPairConnection {
 public:
  //
  // Construct from 2 ApvConnection
  //
  ApvPairConnection(ApvConnection apv1, ApvConnection apv2);
  //
  // get methods
  //
  int getFecSlot() const {return fecSlot;}
  int getRingSlot() const {return ringSlot;}
  int getCcuAddress() const {return ccuAddress;}
  int getI2CChannel() const {return i2cChannel;}
  int getI2CAddressApv1() const {return i2cAddressApv1;}
  int getI2CAddressApv2() const {return i2cAddressApv2;}

  ApvConnection getApv1(){return apv1;}
  ApvConnection getApv2(){return apv2;}

  //
  // set methods
  //
  void setFecSlot(int a) {fecSlot = a;}
  void setRingSlot(int a) {ringSlot = a;}
  void setCcuAddress(int a) {ccuAddress = a;}
  void setI2CChannel(int a) {i2cChannel = a;}
  void setI2CAddressApv1(int a) {i2cAddressApv1 = a;}
  void setI2CAddressApv2(int a) {i2cAddressApv2 = a;}

  //
  // FedConnection
  //
  void setFedNumber(int o){fedNumber = o;}
  void setFedChannel(int o){fedChannel = o;}
  int getFedNumber(){return fedNumber;}
  int getFedChannel(){return fedChannel;}

  bool consistencyChecks();

 private:
 
  ApvConnection apv1;
  ApvConnection apv2;

  int fecSlot;
  int ringSlot;
  int ccuAddress;
  int i2cChannel;
  int i2cAddressApv1;
  int i2cAddressApv2;

  int fedNumber;
  int fedChannel;
		    
};

#endif

