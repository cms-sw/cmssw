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

 private:
  unsigned fw_version, a, b, c;
};

#endif
