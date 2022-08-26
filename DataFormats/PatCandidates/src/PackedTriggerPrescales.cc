#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/Common/interface/RefProd.h"

pat::PackedTriggerPrescales::PackedTriggerPrescales(const edm::Handle<edm::TriggerResults> &handle)
    : prescaleValues_(), triggerResults_(edm::RefProd<edm::TriggerResults>(handle).refCore()), triggerNames_(nullptr) {
  prescaleValues_.resize(handle->size(), 0);
}

void pat::PackedTriggerPrescales::addPrescaledTrigger(int index, double prescale) {
  if (unsigned(index) >= triggerResults().size())
    throw cms::Exception("InvalidReference", "Index out of bounds");
  prescaleValues_[index] = prescale;
}
