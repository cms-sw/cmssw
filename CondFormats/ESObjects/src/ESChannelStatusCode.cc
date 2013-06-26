#include "CondFormats/ESObjects/interface/ESChannelStatusCode.h"

ESChannelStatusCode::ESChannelStatusCode() {
  status_ = 0;
}

ESChannelStatusCode::ESChannelStatusCode(const ESChannelStatusCode & ratio) {
  status_ = ratio.status_;
}

ESChannelStatusCode::~ESChannelStatusCode() {
}

ESChannelStatusCode& ESChannelStatusCode::operator=(const ESChannelStatusCode& rhs) {
  status_ = rhs.status_;
  return *this;
}
