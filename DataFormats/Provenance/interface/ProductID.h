#ifndef DataFormats_Provenance_ProductID_h
#define DataFormats_Provenance_ProductID_h

/*----------------------------------------------------------------------
  
ProductID: A unique identifier for each EDProduct within a process.
Used only in Ref, Ptr, and similar classes.

The high order 16 bits is the process index, identifying the process
in which the product was created.  Exception: An index of 0 means that
the product was created prior to the new format (i.e. prior to CMSSW_3_0_0.

The low order 16 bits is the product index, identifying the product that
in which the product was created.  An index of zero means no product.


The 

----------------------------------------------------------------------*/

#include <iosfwd>

namespace edm {

  typedef unsigned short ProcessIndex;
  typedef unsigned short ProductIndex;
  class ProductID {
  public:
    ProductID() : processIndex_(0),
		  productIndex_(0) {}
    explicit
    ProductID(ProductIndex productIndex) : processIndex_(0), productIndex_(productIndex) {}
    ProductID(ProcessIndex processIndex, ProductIndex productIndex) :
      processIndex_(processIndex), productIndex_(productIndex) {}
    bool isValid() const {return productIndex_ != 0;}
    ProcessIndex processIndex() const {return processIndex_;}
    ProcessIndex productIndex() const {return productIndex_;}
    ProductIndex id() const {return productIndex_;} // backward compatibility
    void reset() {processIndex_ = productIndex_ = 0;}

    void swap(ProductID& other);

  private:
    ProcessIndex processIndex_;
    ProductIndex productIndex_;
  };

  inline
  void swap(ProductID& a, ProductID& b) {
    a.swap(b);
  }

  inline
  bool operator==(ProductID const& lh, ProductID const& rh) {
    return lh.processIndex() == rh.processIndex() && lh.productIndex() == rh.productIndex();
  }
  inline
  bool operator!=(ProductID const& lh, ProductID const& rh) {
    return !(lh == rh);
  }

  bool operator<(ProductID const& lh, ProductID const& rh);

  std::ostream&
  operator<<(std::ostream& os, ProductID const& id);
}
#endif
