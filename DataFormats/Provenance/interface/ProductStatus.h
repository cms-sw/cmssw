#ifndef DataFormats_Provenance_ProductStatus_h
#define DataFormats_Provenance_ProductStatus_h

/*----------------------------------------------------------------------
  
ProductStatus: 

$Id: ProductStatus.h,v 1.2 2008/02/02 00:41:01 wmtan Exp $
----------------------------------------------------------------------*/
/*
  ProductStatus
*/
#include <vector>

namespace edm {
  typedef unsigned char byte_t;
  typedef byte_t ProductStatus;
  typedef std::vector<ProductStatus> ProductStatusVector;
  namespace productstatus {
    inline ProductStatus present() {return 0x0;} // Product was made successfully
    inline ProductStatus neverCreated() {return 0x1;} // Product was not made successfully
    inline ProductStatus onDemand() {return 0xfd;} // Product scheduled for on demand production but not made (yet)
    inline ProductStatus unknown() {return 0xfe;} // Status unknown (used for backward compatibility)
    inline ProductStatus invalid() {return 0xff;} // No product (placeholder)
    inline bool present(ProductStatus status) {return status == present();}
    inline bool neverCreated(ProductStatus status) {return status == neverCreated();}
    inline bool onDemand(ProductStatus status) {return status == onDemand();}
    inline bool unknown(ProductStatus status) {return status == unknown();}
    inline bool invalid(ProductStatus status) {return status == invalid();}
  }
}
#endif
