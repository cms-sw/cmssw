#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"

using namespace l1t;


const PrescalesVetosHelper *  PrescalesVetosHelper::readFromEventSetup(const L1TGlobalPrescalesVetos * es){
  return new PrescalesVetosHelper(es);
}

PrescalesVetosHelper *  PrescalesVetosHelper::readAndWriteFromEventSetup(const L1TGlobalPrescalesVetos * es){
  PrescalesVetosHelper * x = new PrescalesVetosHelper(es);
  x->useCopy();
  return x;
}

PrescalesVetosHelper::PrescalesVetosHelper(L1TGlobalPrescalesVetos * w) {
  write_ = w; 
  check_write(); 
  we_own_write_ = false;
  write_->version_ = VERSION_; 
  read_ = write_; 
}

PrescalesVetosHelper::PrescalesVetosHelper(const L1TGlobalPrescalesVetos * es) {read_ = es; write_=NULL;}

void PrescalesVetosHelper::useCopy(){
  write_ = new L1TGlobalPrescalesVetos(*read_);
  we_own_write_ = true;
  read_  = write_;
}

PrescalesVetosHelper::~PrescalesVetosHelper() {
  if (we_own_write_ && write_) delete write_;
}
