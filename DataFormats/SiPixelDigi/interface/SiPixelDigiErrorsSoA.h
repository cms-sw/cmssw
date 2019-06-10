#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSoA_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigiErrorsSoA_h

#include "DataFormats/SiPixelDigi/interface/PixelErrors.h"

#include <cstdint>
#include <vector>

class SiPixelDigiErrorsSoA {
public:
  SiPixelDigiErrorsSoA() = default;
  explicit SiPixelDigiErrorsSoA(size_t nErrors, const PixelErrorCompact *error, const PixelFormatterErrors *err);
  ~SiPixelDigiErrorsSoA() = default;

  auto size() const { return error_.size(); }

  const PixelFormatterErrors *formatterErrors() const { return formatterErrors_; }

  const PixelErrorCompact& error(size_t i) const { return error_[i]; }
  
  const std::vector<PixelErrorCompact>& errorVector() const { return error_; }
  
private:
  std::vector<PixelErrorCompact> error_;
  const PixelFormatterErrors *formatterErrors_ = nullptr;
};

#endif
