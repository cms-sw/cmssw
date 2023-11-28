#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

#include <iostream>
#include <iomanip>

OpticalAlignParam::OpticalAlignParam() {
  quality_ = -1;
  dim_type_ = "";
}

std::ostream& operator<<(std::ostream& os, const OpticalAlignInfo& r) {
  os << "Name: " << r.name_ << std::endl;
  os << "Parent Name: " << r.parentName_ << std::endl;
  os << "Type: " << r.type_ << "  ID: " << r.ID_ << std::endl;
  int iw = os.width();      // save current width
  int ip = os.precision();  // save current precision
  int now = 12;
  int nop = 5;
  os << std::setw(now) << std::setprecision(nop) << "member";
  os << std::setw(now) << std::setprecision(nop) << "dim_type";
  os << std::setw(now) << std::setprecision(nop) << "value";
  os << std::setw(now) << std::setprecision(nop) << "error";
  os << std::setw(now) << std::setprecision(nop) << "quality" << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.x_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.y_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.z_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.angx_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.angy_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << r.angz_ << std::endl;
  os << std::setw(now) << std::setprecision(nop) << "--- Extra Entries --- " << std::endl;
  size_t max = r.extraEntries_.size();
  size_t iE = 0;
  while (iE < max) {
    os << "[" << iE << "]" << r.extraEntries_[iE];
    iE++;
  }
  os << std::setprecision(ip) << std::setw(iw);
  return os;
}

std::ostream& operator<<(std::ostream& os, const OpticalAlignParam& r) {
  int iw = std::cout.width();      // save current width
  int ip = std::cout.precision();  // save current precision
  int now = 12;
  int nop = 5;
  os << std::setw(now) << std::setprecision(nop) << r.name_;
  os << std::setw(now) << std::setprecision(nop) << r.dim_type_;
  os << std::setw(now) << std::setprecision(nop) << r.value_;
  os << std::setw(now) << std::setprecision(nop) << r.error_;
  os << std::setw(now) << std::setprecision(nop) << r.quality_ << std::endl;

  // Reset the values we changed
  std::cout << std::setprecision(ip) << std::setw(iw);
  return os;
}

OpticalAlignParam* OpticalAlignInfo::findExtraEntry(std::string& name) {
  OpticalAlignParam* param = nullptr;
  std::vector<OpticalAlignParam>::iterator ite;
  for (ite = extraEntries_.begin(); ite != extraEntries_.end(); ite++) {
    if ((*ite).name_ == name) {
      param = &(*ite);
      break;
    }
  }
  return param;
}
