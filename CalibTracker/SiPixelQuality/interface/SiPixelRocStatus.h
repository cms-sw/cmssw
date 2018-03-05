#ifndef SIPIXELROCSTATUS_h
#define SIPIXELROCSTATUS_h

#include <ctime>

// ----------------------------------------------------------------------
class SiPixelRocStatus {
public:
  SiPixelRocStatus();
  ~SiPixelRocStatus();
  void fillDIGI(int idc);
  void updateDIGI(int idc, unsigned long int hits);

  void fillStuckTBM(unsigned int fed, unsigned int link, std::time_t time);
  void updateStuckTBM(unsigned int fed, unsigned int link, std::time_t time, unsigned long int freq);

  // stuckTBM
  bool isStuckTBM(){ return isStuckTBM_; }
  unsigned int getBadFed(){ return badFed_; }
  unsigned int getBadLink(){ return badLink_; }
  std::time_t getStartBadTime(){ return startBadTime_; }
  unsigned long int getBadFreq(){ return badFreq_; }

  // occpancy
  unsigned long int digiOccDC(int idc);
  unsigned long int digiOccROC();

  int nDC(){ return nDC_;}
   

private:
  const int nDC_ = 26;
  unsigned long int fDC[26];

  bool isStuckTBM_;
  unsigned int badFed_;
  unsigned int badLink_;
  std::time_t startBadTime_;
  unsigned long int badFreq_;
 
};

#endif
