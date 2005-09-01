#ifndef Framework_GROUP_h
#define Framework_GROUP_h

/*----------------------------------------------------------------------
  
Group: A collection of information related to a single EDProduct. This
is the storage unit of such information.

$Id: Group.h,v 1.5 2005/07/30 04:43:27 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Provenance.h"

namespace edm {
  class Group {
  public:
    explicit Group(std::auto_ptr<Provenance> prov);

    Group(std::auto_ptr<EDProduct> edp,
	  std::auto_ptr<Provenance> prov,
	  bool acc=true);

    ~Group();

    void swap(Group& other);
      
    // drop (hide) on input: means hide, but provenance is available
    // drop (write) on output: choose not to write, output module
    //   still chooses on a per product to include or exclude
    //   the provenance

    // these and not hidden and data will be populated if retrieved
    bool isAccessible() const;

    EDProduct const* product() const { return product_; }

    Provenance const& provenance() const { return *provenance_; }

    ProductDescription const& productDescription() const { return provenance_->product; }

    void setID(ProductID id);

    // The following is const because we can add an EDProduct to the
    // cache after creation of the Group, without changing the meaning
    // of the Group.
    void setProduct(std::auto_ptr<EDProduct> prod) const;

    // Write the group to the stream.
    void write(std::ostream& os) const;

  private:
    Group(const Group&);
    void operator=(const Group&);

    mutable EDProduct*   product_;
    Provenance*          provenance_;
    bool                 accessible_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, Group const& g) {
    g.write(os);
    return os;
  }

}
#endif //  Framework_GROUP_h
