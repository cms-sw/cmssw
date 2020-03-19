#include "CalibFormats/SiPixelObjects/interface/PixelFEDParameters.h"
#include <ostream>

using namespace pos;

PixelFEDParameters::PixelFEDParameters() {
  fednumber_ = 0;
  crate_ = 0;
  vmebaseaddress_ = 0;
}

PixelFEDParameters::~PixelFEDParameters() {}

unsigned int PixelFEDParameters::getFEDNumber() const { return fednumber_; }

unsigned int PixelFEDParameters::getCrate() const { return crate_; }

unsigned int PixelFEDParameters::getVMEBaseAddress() const { return vmebaseaddress_; }

void PixelFEDParameters::setFEDParameters(unsigned int fednumber, unsigned int crate, unsigned int vmebaseaddress) {
  fednumber_ = fednumber;
  crate_ = crate;
  vmebaseaddress_ = vmebaseaddress;
}

void PixelFEDParameters::setFEDNumber(unsigned int fednumber) { fednumber_ = fednumber; }

void PixelFEDParameters::setCrate(unsigned int crate) { crate_ = crate; }

void PixelFEDParameters::setVMEBaseAddress(unsigned int vmebaseaddress) { vmebaseaddress_ = vmebaseaddress; }

std::ostream& pos::operator<<(std::ostream& s, const PixelFEDParameters& pFEDp) {
  s << "FED Number:" << pFEDp.fednumber_ << std::endl;
  s << "Crate Number:" << pFEDp.crate_ << std::endl;
  s << "VME Base Address:" << pFEDp.vmebaseaddress_ << std::endl;

  return s;
}
