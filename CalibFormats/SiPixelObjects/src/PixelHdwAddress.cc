//
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"
#include <string>
#include <cassert>
#include <ostream>
#include <iostream>

using namespace pos;

//====================================================================================
PixelHdwAddress::PixelHdwAddress()
    : mfec_(0),
      mfecchannel_(0),
      portaddress_(0),
      hubaddress_(0),
      rocid_(0),
      fednumber_(0),
      fedchannel_(0),
      fedrocnumber_(0) {}

//====================================================================================
PixelHdwAddress::PixelHdwAddress(int fecnumber,
                                 int mfec,
                                 int mfecchannel,
                                 int hubaddress,
                                 int portaddress,
                                 int rocid,
                                 int fednumber,
                                 int fedchannel,
                                 int fedrocnumber)
    : fecnumber_(fecnumber),
      mfec_(mfec),
      mfecchannel_(mfecchannel),
      portaddress_(portaddress),
      hubaddress_(hubaddress),
      rocid_(rocid),
      fednumber_(fednumber),
      fedchannel_(fedchannel),
      fedrocnumber_(fedrocnumber) {
  //std::cout << "Created PixelHdwAddress:"<<std::endl;
  //std::cout << *this << std::endl;
}

//====================================================================================
std::ostream& pos::operator<<(std::ostream& s, const PixelHdwAddress& pixelroc) {
  s << "[PixelHdwAddress::operator<<]" << std::endl;
  s << "fecnumber   :" << pixelroc.fecnumber_ << std::endl;
  s << "mfec        :" << pixelroc.mfec_ << std::endl;
  s << "mfecchannel :" << pixelroc.mfecchannel_ << std::endl;
  s << "portaddress :" << pixelroc.portaddress_ << std::endl;
  s << "hubaddress  :" << pixelroc.hubaddress_ << std::endl;
  s << "rocid       :" << pixelroc.rocid_ << std::endl;
  s << "fednumber   :" << pixelroc.fednumber_ << std::endl;
  s << "fedchannel  :" << pixelroc.fedchannel_ << std::endl;
  s << "fedrocnumber:" << pixelroc.fedrocnumber_ << std::endl;

  return s;
}

//====================================================================================
const PixelHdwAddress& PixelHdwAddress::operator=(const PixelHdwAddress& aROC) {
  fecnumber_ = aROC.fecnumber_;
  mfec_ = aROC.mfec_;
  mfecchannel_ = aROC.mfecchannel_;
  portaddress_ = aROC.portaddress_;
  hubaddress_ = aROC.hubaddress_;
  rocid_ = aROC.rocid_;
  fednumber_ = aROC.fednumber_;
  fedchannel_ = aROC.fedchannel_;
  fedrocnumber_ = aROC.fedrocnumber_;

  return *this;
}

//====================================================================================
// Added by Dario
void PixelHdwAddress::setAddress(std::string what, int value) {
  std::string mthn = "[PixelHdwAddress::setAddress()]\t\t\t    ";
  if (what == "fecnumber") {
    fecnumber_ = value;
  } else if (what == "mfec") {
    mfec_ = value;
  } else if (what == "mfecchannel") {
    mfecchannel_ = value;
  } else if (what == "portaddress") {
    portaddress_ = value;
  } else if (what == "hubaddress") {
    hubaddress_ = value;
  } else if (what == "rocid") {
    rocid_ = value;
  } else if (what == "fednumber") {
    fednumber_ = value;
  } else if (what == "fedchannel") {
    fedchannel_ = value;
  } else if (what == "fedrocnumber") {
    fedrocnumber_ = value;
  } else {
    std::cout << __LINE__ << "]\t" << mthn << "Could not set a value for " << what << " (invalid keyword)" << std::endl;
    assert(0);
  }
}

//====================================================================================
void PixelHdwAddress::compare(std::string what, bool& changed, unsigned int newValue, unsigned int& oldValue) {
  std::string mthn = "[PixelHdwAddress::compare()]\t\t\t    ";
  changed = false;
  oldValue = 0;

  if (what == "fecnumber") {
    if (fecnumber_ != newValue) {
      changed = true;
      oldValue = fecnumber_;
      return;
    }
  } else if (what == "mfec") {
    if (mfec_ != newValue) {
      changed = true;
      oldValue = mfec_;
      return;
    }
  } else if (what == "mfecchannel") {
    if (mfecchannel_ != newValue) {
      changed = true;
      oldValue = mfecchannel_;
      return;
    }
  } else if (what == "portaddress") {
    if (portaddress_ != newValue) {
      changed = true;
      oldValue = portaddress_;
      return;
    }
  } else if (what == "hubaddress") {
    if (hubaddress_ != newValue) {
      changed = true;
      oldValue = hubaddress_;
      return;
    }
  } else if (what == "rocid") {
    if (rocid_ != newValue) {
      changed = true;
      oldValue = rocid_;
      return;
    }
  } else if (what == "fednumber") {
    if (fednumber_ != newValue) {
      changed = true;
      oldValue = fednumber_;
      return;
    }
  } else if (what == "fedchannel") {
    if (fedchannel_ != newValue) {
      changed = true;
      oldValue = fedchannel_;
      return;
    }
  } else if (what == "fedrocnumber") {
    if (fedrocnumber_ != newValue) {
      changed = true;
      oldValue = fedrocnumber_;
      return;
    }
  } else {
    std::cout << __LINE__ << "]\t" << mthn << "Could not compare value for " << what << " (invalid keyword)"
              << std::endl;
    assert(0);
  }
}

//====================================================================================
bool PixelHdwAddress::operator()(const PixelHdwAddress& roc1, const PixelHdwAddress& roc2) const {
  if (roc1.fednumber_ < roc2.fednumber_)
    return true;
  if (roc1.fednumber_ > roc2.fednumber_)
    return false;
  if (roc1.fedchannel_ < roc2.fedchannel_)
    return true;
  if (roc1.fedchannel_ > roc2.fedchannel_)
    return false;

  return (roc1.fedrocnumber_ < roc2.fedrocnumber_);
}
