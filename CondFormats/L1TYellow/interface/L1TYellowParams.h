#ifndef L1TYELLOWPARAMS_H
#define L1TYELLOWPARAMS_H

class L1TYellowParams {

 public:
  // constructor
  L1TYellowParams() {}

  unsigned firmwareVersion() const {return fw_version;}
  unsigned paramA() const {return a;}
  unsigned paramB() const {return b;}
  unsigned paramC() const {return c;}

  void setFirmwareVersion(unsigned v) {fw_version=v;};
  void setParamA(unsigned v) {a = v;}
  void setParamB(unsigned v) {b = v;}
  void setParamC(unsigned v) {c = v;}

 private:
  unsigned fw_version, a, b, c;
};

#endif
