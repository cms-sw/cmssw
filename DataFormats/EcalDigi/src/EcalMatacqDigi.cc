// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
// $Id: EcalMatacqDigi.cc,v 1.6 2011/08/30 18:42:58 wmtan Exp $
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"

using namespace std;

const double EcalMatacqDigi::lsb_ = 0.25e-3;// in Volt

#if 0
void EcalMatacqDigi::setSize(const int& size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}
#endif
  
std::ostream& operator<<(std::ostream& s, const EcalMatacqDigi& digi) {
  s << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++){
    s << "  " << digi.amplitudeV(i) << std::endl;
  }
  return s;
}

void EcalMatacqDigi::swap(EcalMatacqDigi& a){
  data_.swap(a.data_);
  std::swap(chId_, a.chId_);
  std::swap(ts_, a.ts_);
  std::swap(tTrigS_, a.tTrigS_);
  std::swap(version_, a.version_);
#if (ECAL_MATACQ_DIGI_VERS>=2)
  std::swap(bxId_, a.bxId_);
  std::swap(l1a_, a.l1a_);
  std::swap(triggerType_, a.triggerType_);
  std::swap(orbitId_, a.orbitId_);
  std::swap(trigRec_, a.trigRec_);
  std::swap(postTrig_, a.postTrig_);
  std::swap(vernier_, a.vernier_);
  std::swap(delayA_, a.delayA_);
  std::swap(emtcDelay_, a.emtcDelay_);
  std::swap(emtcPhase_, a.emtcPhase_);
  std::swap(attenuation_dB_, a.attenuation_dB_);
  std::swap(laserPower_, a.laserPower_);
  std::swap(tv_sec_, a.tv_sec_);
  std::swap(tv_usec_, a.tv_usec_);
#endif
}
