#ifndef SIPIXELROCSTATUS_h
#define SIPIXELROCSTATUS_h

#include <ctime>

// ----------------------------------------------------------------------
class SiPixelRocStatus {
public:
  SiPixelRocStatus();
  ~SiPixelRocStatus();
  void fillDIGI();
  void updateDIGI(unsigned int hits);
  void fillStuckTBM();

  // stuckTBM
  bool isStuckTBM(){ return isStuckTBM_; }

  // occpancy
  unsigned int digiOccROC();

private:

  unsigned int fDC;
  bool isStuckTBM_;
 
};

#endif
