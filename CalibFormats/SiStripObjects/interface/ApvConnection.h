#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_APVCONNECTION_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_APVCONNECTION_H

class ApvConnection {
 public:
  ApvConnection();
  ApvConnection(int, int, int, int, int);

  //
  // get methods
  //

  int getFecSlot() const {return fecSlot;}
  int getRingSlot() const {return ringSlot;}
  int getCcuAddress() const {return ccuAddress;}
  int getI2CChannel() const {return i2cChannel;}
  int getI2CAddress() const {return i2cAddress;}

  //
  // set methods
  //  

  void setFecSlot(int a) {fecSlot = a;}
  void setRingSlot(int a) {ringSlot = a;}
  void setCcuAddress(int a) {ccuAddress = a;}
  void setI2CChannel(int a) {i2cChannel = a;}
  void setI2CAddress(int a) {i2cAddress = a;}
  
  //
  // Equaity Operator
  //
  bool operator==(const ApvConnection&);
 private:
  int fecSlot;
  int ringSlot;
  int ccuAddress;
  int i2cChannel;
  int i2cAddress;

};

#endif
