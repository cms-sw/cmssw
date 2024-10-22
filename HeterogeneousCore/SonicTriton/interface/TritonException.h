#ifndef HeterogeneousCore_SonicTriton_TritonException
#define HeterogeneousCore_SonicTriton_TritonException

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class TritonException : public cms::Exception {
public:
  explicit TritonException(std::string const& aCategory, bool signal = false);
  void convertToWarning() const;
};

#endif
