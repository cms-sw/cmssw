#include "CondFormats/Calibration/interface/BlobNoises.h"
BlobNoises::BlobNoises(){}
BlobNoises::~BlobNoises(){}
void BlobNoises::fill(unsigned int id, short bsize){
  //short id_s = (short)id;
  for(short i=0;i<bsize;i++) v_noises.push_back(i+1);
  for(unsigned int i=0;i<id+1;i++) {
    DetRegistry reg;
    reg.detid = i;
    reg.ibegin = i;
    reg.iend = i;
    indexes.push_back(reg);
  }
}
