#include "L1Trigger/L1TMuonEndCap/interface/EndCapParamsHelper.h"

#include <iostream>

using namespace l1t;
using namespace std;

const EndCapParamsHelper *  EndCapParamsHelper::readFromEventSetup(const L1TMuonEndCapParams * es){
  return new EndCapParamsHelper(es);
}

EndCapParamsHelper *  EndCapParamsHelper::readAndWriteFromEventSetup(const L1TMuonEndCapParams * es){
  EndCapParamsHelper * x = new EndCapParamsHelper(es);
  x->useCopy();
  return x;
}

EndCapParamsHelper::EndCapParamsHelper(L1TMuonEndCapParams * w) {
  write_ = w;
  check_write();
  we_own_write_ = false;
  //write_->m_version = VERSION;
  read_ = write_;
}

EndCapParamsHelper::EndCapParamsHelper(const L1TMuonEndCapParams * es) {read_ = es; write_=nullptr;}

void EndCapParamsHelper::useCopy(){
  write_ = new L1TMuonEndCapParams(*read_);
  we_own_write_ = true;
  read_  = write_;
}

EndCapParamsHelper::~EndCapParamsHelper() {
  if (we_own_write_ && write_) delete write_;
}


// print all the L1 GT stable parameters
void EndCapParamsHelper::print(std::ostream& myStr) const {
    myStr << "\nL1T EndCap  Parameters \n" << std::endl;
}
