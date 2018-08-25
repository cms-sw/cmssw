#ifndef DataFormats_MTDDetId_MTDDetId_h
#define DataFormats_MTDDetId_MTDDetId_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include <ostream>

/** 
    @class MTDDetId
    @brief Detector identifier base class for the MIP Timing Layer.

    bit 31-28: Detector Forward
    bit 27-25: Subdetctor FastTiming

    bit 24-23: MTD subdetector BTL/ETL
    bit 22   : side (positive = 1, negative = 0)
    bit 21-16: rod/ring sequential number

*/

class MTDDetId : public DetId {
  
 protected:

  /** Enumerated type for Forward sub-deteector systems. */
  enum SubDetector { subUNKNOWN=0, FastTime=1 };
  
  /** Enumerated type for MTD sub-deteector systems. */
  enum MTDType { typeUNKNOWN=0, BTL=1, ETL=2 };
  
  static const uint32_t kMTDsubdOffset             = 23;
  static const uint32_t kMTDsubdMask               = 0x3;
  static const uint32_t kZsideOffset               = 22;
  static const uint32_t kZsideMask                 = 0x1;
  static const uint32_t kRodRingOffset             = 16;
  static const uint32_t kRodRingMask               = 0x3F;  

 public:
  
  // ---------- Constructors, enumerated types ----------
  
  /** Construct a null id */
 MTDDetId()  : DetId() {;}
  
  /** Construct from a raw value */
 MTDDetId( const uint32_t& raw_id ) : DetId( raw_id ) {;}
  
  /** Construct from generic DetId */
 MTDDetId( const DetId& det_id )  : DetId( det_id.rawId() ) {;}
  
  /** Construct and fill only the det and sub-det fields. */
 MTDDetId( Detector det, int subdet ) : DetId( det, subdet ) {;}
  
  // ---------- Common methods ----------
  
  /** Returns enumerated type specifying MTD sub-detector. */
  inline SubDetector subDetector() const { return static_cast<MTDDetId::SubDetector>(subdetId()); }
  
  /** Returns enumerated type specifying MTD sub-detector, i.e. BTL or ETL. */
  inline int mtdSubDetector() const { return (id_>>kMTDsubdOffset)&kMTDsubdMask; }

  /** Returns MTD side, i.e. Z-=0 or Z+=1. */
  inline int mtdSide() const { return (id_>>kZsideOffset)&kZsideMask; }
  /** Return MTD side, i.e. Z-=-1 or Z+=1. */
  inline int zside() const { return (mtdSide()==1 ? (1):(-1)) ; }
  
  /** Returns MTD rod/ring number. */
  inline int mtdRR() const { return (id_>>kRodRingOffset)&kRodRingMask; }
  
};

std::ostream& operator<< ( std::ostream&, const MTDDetId& );

#endif // DataFormats_MTDDetId_MTDDetId_h

