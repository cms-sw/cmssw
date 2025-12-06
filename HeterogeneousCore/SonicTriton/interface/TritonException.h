#ifndef HeterogeneousCore_SonicTriton_TritonException
#define HeterogeneousCore_SonicTriton_TritonException

#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class TritonException : public cms::Exception {
public:
  explicit TritonException(std::string const& aCategory, const edm::ServiceToken* token = nullptr);
  void convertToWarning() const;
};

#endif
