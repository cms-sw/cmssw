#include "HeterogeneousCore/SonicTriton/interface/TritonRetryActionBase.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "FWCore/Utilities/interface/Exception.h"

// Deliberately not done in the constructor: retry actions are constructed from within their
// owning client's SonicClientBase base-class subobject constructor (see
// SonicClientBase::SonicClientBase()), before the most-derived TritonClient subobject exists.
// dynamic_cast during construction reflects the object's dynamic type as constructed so far,
// which at that point is only SonicClientBase -- a cast attempted there would incorrectly fail
// for every client, not just misconfigured ones. Calling this lazily, from retry(), avoids that:
// by then the owning client is fully constructed.
TritonClient& TritonRetryActionBase::tritonClient() const {
  auto* tc = dynamic_cast<TritonClient*>(&client_);
  if (!tc) {
    throw cms::Exception("Configuration") << "TritonRetryActionBase: client is not a TritonClient. This retry "
                                             "action can only be used with a SonicTriton client.";
  }
  return *tc;
}
