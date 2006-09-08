// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
// $Id$
#include <algorithm>
#include "DataFormats/EcalDigi/interface/EcalMatacqDigi.h"

using namespace std;

const double EcalMatacqDigi::lsb_ = 0.25e-3;//V

#if 0
void EcalMatacqDigi::setSize(int size) {
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
}
