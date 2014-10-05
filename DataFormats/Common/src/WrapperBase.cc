/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include <cassert>

namespace edm {
  WrapperBase::WrapperBase() : ViewTypeChecker() {}

  WrapperBase::~WrapperBase() {}

  void WrapperBase::fillView(ProductID const& id,
                           std::vector<void const*>& pointers,
                           helper_vector_ptr& helpers) const {
    // This should never be called with non-empty arguments, or an
    // invalid ID; any attempt to do so is an indication of a coding
    // error.
    assert(id.isValid());
    assert(pointers.empty());
    assert(helpers.get() == 0);

    do_fillView(id, pointers, helpers);
  }

  void WrapperBase::setPtr(std::type_info const& iToType,
                         unsigned long iIndex,
                         void const*& oPtr) const {
    do_setPtr(iToType, iIndex, oPtr);
  }

  void
  WrapperBase::fillPtrVector(std::type_info const& iToType,
                              std::vector<unsigned long> const& iIndicies,
                              std::vector<void const*>& oPtr) const {
    do_fillPtrVector(iToType, iIndicies, oPtr);
  }

}
