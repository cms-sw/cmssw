#include "DataFormats/SiPixelDigi/interface/SiPixelDigiErrorsSoA.h"

#include <cassert>

SiPixelDigiErrorsSoA::SiPixelDigiErrorsSoA(size_t nErrors, const PixelErrorCompact *error, const PixelFormatterErrors *err):
  error_(error, error+nErrors),
  formatterErrors_(err)
{
  assert(error_.size() == nErrors);
}
