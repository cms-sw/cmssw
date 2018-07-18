#ifndef SIPIXELROCSTATUS_h
#define SIPIXELROCSTATUS_h

// ----------------------------------------------------------------------
class SiPixelRocStatus {
public:
  SiPixelRocStatus();
  ~SiPixelRocStatus();
  void fillDIGI();
  void updateDIGI(unsigned int hits);
  void fillFEDerror25();

  // stuckTBM
  bool isFEDerror25(){ return isFEDerror25_; }

  // occpancy
  unsigned int digiOccROC();

private:

  unsigned int fDC;
  bool isFEDerror25_;
 
};

#endif
