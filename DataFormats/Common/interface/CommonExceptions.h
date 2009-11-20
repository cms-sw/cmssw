#ifndef DataFormats_Common_CommonExceptions_h
#define DataFormats_Common_CommonExceptions_h
namespace edm {
  class ProductID;
  void checkForWrongProduct(ProductID const& keyID, ProductID const& refID);
}
#endif
