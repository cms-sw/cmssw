#ifndef SIPIXELROCSTATUS_h
#define SIPIXELROCSTATUS_h

// ----------------------------------------------------------------------
class SiPixelRocStatus {
public:
  SiPixelRocStatus();
  ~SiPixelRocStatus();

  void fillDIGI();
  void fillFEDerror25();

  void updateDIGI(unsigned int hits);
  void updateFEDerror25(bool fedError25);

  // occpancy
  unsigned int digiOccROC();
  // FEDerror25 for stuckTBM
  bool isFEDerror25();

private:
  unsigned int fDC_;
  bool isFEDerror25_;
};

#endif
