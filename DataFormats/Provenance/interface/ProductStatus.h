#ifndef DataFormats_Provenance_ProductStatus_h
#define DataFormats_Provenance_ProductStatus_h

/*----------------------------------------------------------------------
  
ProductStatus: 

----------------------------------------------------------------------*/
/*
  ProductStatus
*/

namespace edm {
  typedef unsigned char byte_t;
  typedef byte_t ProductStatus;
  namespace productstatus {
    inline ProductStatus present() {return 0x0;} // Product was made successfully
    inline ProductStatus neverCreated() {return 0x1;} // Product was not made successfully, reason not known.
    // The dropped status is no longer used
    // A product's status is not changed when it is dropped
    // so it may be dropped and have status present or neverCreated or ...
    // Can we delete dropped from this code or is there a backward
    // compatibility issue???
    inline ProductStatus dropped() {return 0x2;}
    inline ProductStatus producerNotRun() {return 0x3;} // Producer was never run
    inline ProductStatus producerNotCompleted() {return 0x4;} // Producer was run but did not finish (i.e. threw)
    inline ProductStatus producerDidNotPutProduct() {return 0x5;} // Producer did not put product
    inline ProductStatus unscheduledProducerNotRun() {return 0x6;} // Producer was never run
    inline ProductStatus unknown() {return 0xfe;} // Status unknown 
    inline ProductStatus uninitialized() {return 0xff;} // Status not yet set
    inline bool present(ProductStatus status) {return status == present();}
    inline bool neverCreated(ProductStatus status) {return status == neverCreated();}
    inline bool dropped(ProductStatus status) {return status == dropped();}
    inline bool producerNotRun(ProductStatus status) {return status == producerNotRun();}
    inline bool producerNotCompleted(ProductStatus status) {return status == producerNotCompleted();}
    inline bool producerDidNotPutProduct(ProductStatus status) {return status == producerDidNotPutProduct();}
    inline bool unscheduledProducerNotRun(ProductStatus status) {return status == unscheduledProducerNotRun();}
    inline bool unknown(ProductStatus status) {return status == unknown();}
    inline bool uninitialized(ProductStatus status) {return status == uninitialized();}

    inline bool presenceUnknown(ProductStatus status) {return uninitialized(status) || unknown(status);}
    inline bool notPresent(ProductStatus status) {return !present(status) && !presenceUnknown(status);}
  }
}
#endif
