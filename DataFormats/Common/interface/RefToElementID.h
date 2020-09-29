#ifndef DataFormats_Common_RefToElementID_h
#define DataFormats_Common_RefToElementID_h

#include "DataFormats/Provenance/interface/ElementID.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {

  template <class C>
  edm::ElementID refToElementID(const edm::Ref<C>& ref) {
    return edm::ElementID(ref.id(), ref.index());
  }

  template <class C>
  edm::ElementID refToElementID(const edm::RefToBase<C>& ref) {
    return edm::ElementID(ref.id(), ref.key());
  }
}  // namespace edm

#endif