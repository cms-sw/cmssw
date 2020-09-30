#ifndef DataFormats_Common_refToElementID_h
#define DataFormats_Common_refToElementID_h

#include "DataFormats/Provenance/interface/ElementID.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {

  template <typename C, typename T, typename F>
  edm::ElementID refToElementID(const edm::Ref<C, T, F>& ref) {
    return edm::ElementID(ref.id(), ref.index());
  }

  template <typename C>
  edm::ElementID refToElementID(const edm::RefToBase<C>& ref) {
    return edm::ElementID(ref.id(), ref.key());
  }
}  // namespace edm

#endif