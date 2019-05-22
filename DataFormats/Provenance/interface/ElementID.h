#ifndef DataFormats_Provenance_ElementID_h
#define DataFormats_Provenance_ElementID_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include <iosfwd>

namespace edm {
  /**
   * ElementID is a unique identifier for an element within a
   * container. It extends the ProductID concept by adding an index to
   * an object within a container.
   *
   * It provides both index() and key() methods so that it can be used
   * in place of Ref/Ptr in the interfaces of e.g. ValueMap or Association.
   */
  class ElementID {
  public:
    using key_type = unsigned int;

    ElementID() = default;
    explicit ElementID(edm::ProductID id, key_type ind) : index_(ind), id_(id) {}

    bool isValid() const { return id_.isValid(); }
    ProductID id() const { return id_; }
    key_type index() const { return index_; }
    key_type key() const { return index_; }
    void reset() {
      index_ = 0;
      id_.reset();
    }

    void swap(ElementID& other);

  private:
    key_type index_ = 0;
    ProductID id_;
  };

  inline void swap(ElementID& a, ElementID& b) { a.swap(b); }

  inline bool operator==(ElementID const& lh, ElementID const& rh) {
    return lh.index() == rh.index() && lh.id() == rh.id();
  }

  inline bool operator!=(ElementID const& lh, ElementID const& rh) { return !(lh == rh); }

  bool operator<(ElementID const& lh, ElementID const& rh);

  std::ostream& operator<<(std::ostream& os, ElementID const& id);
}  // namespace edm

#endif
