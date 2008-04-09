#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

extern "C" {
  void hwaend_() {return;}

  void cmsending_(int* ecode) {
    edm::LogError("")<<"   ERROR: Herwig stoped run after receiving error code "<<*ecode<<".\n";
    throw cms::Exception("Herwig6Error") <<" Herwig stoped run with error code "<<*ecode<<".";
  }
}
