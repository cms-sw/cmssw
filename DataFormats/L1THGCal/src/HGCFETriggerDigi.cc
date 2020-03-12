#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"

using namespace l1t;

void HGCFETriggerDigi::print(std::ostream& out) const {
  out << "Codec type: " << static_cast<unsigned>(codec_) << std::endl;
  out << "Raw data payload: ";
  for (unsigned i = data_.size(); i > 0; --i) {
    out << (unsigned)data_[i - 1];
  }
  out << std::endl;
}
