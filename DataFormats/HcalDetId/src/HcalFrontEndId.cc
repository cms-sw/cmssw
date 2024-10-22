#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cstdlib>

HcalFrontEndId::HcalFrontEndId(
    const std::string& rbx, int rm, int pixel, int rmfiber, int fiberchannel, int qie, int adc) {
  hcalFrontEndId_ = 0;

  if (rbx.size() < 5)
    return;
  if (rm < 1 || rm > 5)
    return;  //changed to 5 to incorporate CALIB channels which define RM = 5
  if (pixel < 0 || pixel > 19)
    return;
  if (rmfiber < 1 || rmfiber > 8)
    return;
  if (fiberchannel < 0 || fiberchannel > 2)
    return;
  if (qie < 1 || qie > 4)
    return;
  if (adc < 0 || adc > 5)
    return;

  int num = -1;
  if (!rbx.compare(0, 3, "HBM")) {
    num = 0 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HBP")) {
    num = 18 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HEM")) {
    num = 18 * 2 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HEP")) {
    num = 18 * 3 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 4, "HO2M")) {
    num = 18 * 4 + atoi(rbx.substr(4, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 4, "HO1M")) {
    num = 18 * 4 + 12 + atoi(rbx.substr(4, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HO0")) {
    num = 18 * 4 + 12 * 2 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 4, "HO1P")) {
    num = 18 * 4 + 12 * 3 + atoi(rbx.substr(4, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 4, "HO2P")) {
    num = 18 * 4 + 12 * 4 + atoi(rbx.substr(4, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HFM")) {
    num = 18 * 4 + 12 * 5 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else if (!rbx.compare(0, 3, "HFP")) {
    num = 18 * 4 + 12 * 6 + atoi(rbx.substr(3, 2).c_str()) - 1;
  } else
    return;

  hcalFrontEndId_ |= ((adc + 1) & 0x7);
  hcalFrontEndId_ |= ((qie - 1) & 0x3) << 3;
  hcalFrontEndId_ |= (fiberchannel & 0x3) << 5;
  hcalFrontEndId_ |= ((rmfiber - 1) & 0x7) << 7;
  hcalFrontEndId_ |= (pixel & 0x1F) << 10;
  hcalFrontEndId_ |= ((rm - 1) & 0x7) << 15;
  hcalFrontEndId_ |= (num & 0xFF) << 18;
}

HcalFrontEndId::~HcalFrontEndId() {}

std::string HcalFrontEndId::rbx() const {
  std::string subdets[11] = {"HBM", "HBP", "HEM", "HEP", "HO2M", "HO1M", "HO0", "HO1P", "HO2P", "HFM", "HFP"};

  int box = hcalFrontEndId_ >> 18;
  int num = -1;
  int subdet_index = -1;
  if (box < 18 * 4) {
    num = box % 18;
    subdet_index = (box - num) / 18;
  } else {
    num = (box - 18 * 4) % 12;
    subdet_index = 4 + (box - 18 * 4 - num) / 12;
  }
  std::stringstream tempss;
  tempss << std::setw(2) << std::setfill('0') << num + 1;
  return subdets[subdet_index] + tempss.str();
}

std::ostream& operator<<(std::ostream& s, const HcalFrontEndId& id) {
  return s << id.rbx() << id.rm() << '[' << id.rmFiber() << '/' << id.fiberChannel() << "] pix=" << id.pixel()
           << " qiecard=" << id.qieCard() << " adc=" << id.adc();
}
