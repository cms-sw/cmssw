#include "L1Trigger/L1TGlobal/interface/PrescalesVetosFractHelper.h"

using namespace l1t;

const PrescalesVetosFractHelper* PrescalesVetosFractHelper::readFromEventSetup(const L1TGlobalPrescalesVetosFract* es) {
  return new PrescalesVetosFractHelper(es);
}

PrescalesVetosFractHelper* PrescalesVetosFractHelper::readAndWriteFromEventSetup(
    const L1TGlobalPrescalesVetosFract* es) {
  PrescalesVetosFractHelper* x = new PrescalesVetosFractHelper(es);
  x->useCopy();
  return x;
}

PrescalesVetosFractHelper::PrescalesVetosFractHelper(L1TGlobalPrescalesVetosFract* w) {
  write_ = w;
  check_write();
  we_own_write_ = false;
  write_->version_ = VERSION_;
  read_ = write_;
}

PrescalesVetosFractHelper::PrescalesVetosFractHelper(const L1TGlobalPrescalesVetosFract* es) {
  read_ = es;
  write_ = nullptr;
}

void PrescalesVetosFractHelper::useCopy() {
  write_ = new L1TGlobalPrescalesVetosFract(*read_);
  we_own_write_ = true;
  read_ = write_;
}

PrescalesVetosFractHelper::~PrescalesVetosFractHelper() {
  if (we_own_write_ && write_)
    delete write_;
}
