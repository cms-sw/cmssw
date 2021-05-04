//      ====================================================================
//
//      Guid.cpp
//      --------------------------------------------------------------------
//
//      Package   : Persistent Guid to identify objects in the persistent
//              world.
//
//      Author    : Markus Frank
//
//      ====================================================================
#include "Guid.h"
#include <cstring>
#include <cassert>

namespace edm {
  /// Initialize a new Guid
  void Guid::init(bool usetime) {
    if (usetime) {
      ::uuid_generate_time(data_);
    } else {
      // uuid_generate() defaults to uuid_generate_random() if /dev/urandom
      // is available; if /dev/urandom is not available, then it is better
      // to let uuid_generate() choose the best fallback rather than forcing
      // use of an inferior source of randomness
      ::uuid_generate(data_);
    }
  }

  std::string const Guid::toBinary() const { return std::string(reinterpret_cast<const char*>(data_), sizeof(data_)); }

  Guid const& Guid::fromBinary(std::string const& source) {
    assert(source.size() == sizeof(data_));
    std::memcpy(data_, source.data(), sizeof(data_));
    return *this;
  }

  std::string const Guid::toString() const {
    char out[UUID_STR_LEN];
    ::uuid_unparse(data_, out);
    return std::string(out);
  }

  // fromString is used only in a unit test, so performance is not critical.
  Guid const& Guid::fromString(std::string const& source) {
    auto err = ::uuid_parse(source.c_str(), data_);
    assert(err == 0);
    return *this;
  }

  bool Guid::operator<(Guid const& g) const { return ::uuid_compare(data_, g.data_) < 0; }
}  // namespace edm
