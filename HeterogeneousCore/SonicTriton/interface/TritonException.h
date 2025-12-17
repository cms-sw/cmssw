#ifndef HeterogeneousCore_SonicTriton_TritonException
#define HeterogeneousCore_SonicTriton_TritonException

#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class TritonService;

class TritonException : public cms::Exception {
public:
  explicit TritonException(std::string const& aCategory, const TritonService* service = nullptr);
  void convertToWarning() const;
};

#endif
