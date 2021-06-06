#ifndef DataFormats_SiPixelDigi_interface_SiPixelErrorsSoA_h
#define DataFormats_SiPixelDigi_interface_SiPixelErrorsSoA_h

#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"

#include <cstdint>
#include <vector>

class SiPixelErrorsSoA {
public:
  SiPixelErrorsSoA() = default;
  explicit SiPixelErrorsSoA(size_t nErrors, const SiPixelErrorCompact *error, const SiPixelFormatterErrors *err)
      : error_(error, error + nErrors), formatterErrors_(err) {}
  ~SiPixelErrorsSoA() = default;

  auto size() const { return error_.size(); }

  const SiPixelFormatterErrors *formatterErrors() const { return formatterErrors_; }

  const SiPixelErrorCompact &error(size_t i) const { return error_[i]; }

  const std::vector<SiPixelErrorCompact> &errorVector() const { return error_; }

private:
  std::vector<SiPixelErrorCompact> error_;
  const SiPixelFormatterErrors *formatterErrors_ = nullptr;
};

#endif
