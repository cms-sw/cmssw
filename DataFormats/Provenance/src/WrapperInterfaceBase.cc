/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include <cassert>

namespace edm {
  WrapperInterfaceBase::WrapperInterfaceBase() {}

  WrapperInterfaceBase::~WrapperInterfaceBase() {}

  void WrapperInterfaceBase::fillView(void const* me,
                                      ProductID const& id,
                                      std::vector<void const*>& pointers,
                                      helper_vector_ptr& helpers) const {
    // This should never be called with non-empty arguments, or an
    // invalid ID; any attempt to do so is an indication of a coding
    // error.
    assert(id.isValid());
    assert(pointers.empty());
    assert(helpers.get() == 0);

    do_fillView(me, id, pointers, helpers);
  }

  void WrapperInterfaceBase::setPtr(void const* me,
                                    std::type_info const& iToType,
                                    unsigned long iIndex,
                                    void const*& oPtr) const {
    do_setPtr(me, iToType, iIndex, oPtr);
  }

  void
  WrapperInterfaceBase::fillPtrVector(void const* me,
                                      std::type_info const& iToType,
                                      std::vector<unsigned long> const& iIndicies,
                                      std::vector<void const*>& oPtr) const {
    do_fillPtrVector(me, iToType, iIndicies, oPtr);
  }
}
