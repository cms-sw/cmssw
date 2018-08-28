#ifndef DataFormats_BTLDetId_BTLDetId_h
#define DataFormats_BTLDetId_BTLDetId_h

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include <ostream>

/** 
    @class BTLDetId
    @brief Detector identifier class for the Barrel Timing Layer.
    The crystal count must start from 0, copy number must be scaled by 1 unit.

    bit 15-10: module sequential number
    bit 9-8  : crystal type (1 - 3)
    bit 7-0  : crystal sequential number within a module ( 0 - 63 )
*/

class BTLDetId : public MTDDetId {
  
 private:
  
  static constexpr uint32_t kBTLmoduleOffset           = 10;
  static constexpr uint32_t kBTLmoduleMask             = 0x3F;
  static constexpr uint32_t kBTLmodTypeOffset          = 8;
  static constexpr uint32_t kBTLmodTypeMask            = 0x3;
  static constexpr uint32_t kBTLCrystalOffset          = 0;
  static constexpr uint32_t kBTLCrystalMask            = 0x3F;

 public:

  /// range constants, need two sets for the time being (one for tiles and one for bars)
  static constexpr int kModulesPerROD = 54;
  static constexpr int kTypeBoundaries[4] = { 0, 18, 36, 54 };
  static constexpr int kCrystalsInPhiTile = 16; // per module and ROD
  static constexpr int kCrystalsInEtaTile = 4; // per module
  static constexpr int kCrystalsInPhiBar = 4; // per module and ROD
  static constexpr int kCrystalsInEtaBar = 16; // per module
  static constexpr int kCrystalsPerROD = kModulesPerROD*kCrystalsInPhiTile*kCrystalsInEtaTile; // 64 crystals per module x 54 modules per rod, independent on geometry scenario Tile or Bar
  static constexpr int MIN_ROD = 1;
  static constexpr int MAX_ROD = 72;
  static constexpr int HALF_ROD = 36;
  static constexpr int MIN_IETA = 1;
  static constexpr int MIN_IPHI = 1;
  static constexpr int MAX_IETA_TILE = kCrystalsInEtaTile*kModulesPerROD;
  static constexpr int MAX_IPHI_TILE = kCrystalsInPhiTile*HALF_ROD;
  static constexpr int MAX_IETA_BAR = kCrystalsInEtaBar*kModulesPerROD;
  static constexpr int MAX_IPHI_BAR = kCrystalsInPhiBar*HALF_ROD;
  static constexpr int MIN_HASH =  0; // always 0 ...
  static constexpr int MAX_HASH =  2*MAX_IPHI_TILE*MAX_IETA_TILE-1; // the total amount is invariant per tile or bar)
  static constexpr int kSizeForDenseIndexing = MAX_HASH + 1 ;

  enum class CrysLayout { tile = 1 , bar = 2 } ;

 public:
  
  // ---------- Constructors, enumerated types ----------
  
  /** Construct a null id */
 BTLDetId() : MTDDetId( DetId::Forward, ForwardSubdetector::FastTime ) { id_ |= ( MTDType::BTL& kMTDsubdMask ) << kMTDsubdOffset ;}
  
  /** Construct from a raw value */
 BTLDetId( const uint32_t& raw_id ) : MTDDetId( raw_id ) {;}
  
  /** Construct from generic DetId */
 BTLDetId( const DetId& det_id )  : MTDDetId( det_id.rawId() ) {;}
  
  /** Construct and fill only the det and sub-det fields. */
 BTLDetId( uint32_t zside, 
           uint32_t rod, 
           uint32_t module, 
           uint32_t modtyp, 
           uint32_t crystal ) : MTDDetId( DetId::Forward, ForwardSubdetector::FastTime ) {
    id_ |= ( MTDType::BTL& kMTDsubdMask ) << kMTDsubdOffset |
      ( zside& kZsideMask ) << kZsideOffset |
      ( rod& kRodRingMask ) << kRodRingOffset |
      ( module& kBTLmoduleMask ) << kBTLmoduleOffset |
      ( modtyp& kBTLmodTypeMask ) << kBTLmodTypeOffset |
      ( (crystal-1)& kBTLCrystalMask ) << kBTLCrystalOffset ; 
  }
  
  // ---------- Common methods ----------
  
  /** Returns BTL module number. */
  inline int module() const { return (id_>>kBTLmoduleOffset)&kBTLmoduleMask; }
  
  /** Returns BTL crystal type number. */
  inline int modType() const { return (id_>>kBTLmodTypeOffset)&kBTLmodTypeMask; }
  
  /** Returns BTL crystal number. */
  inline int crystal() const { return ((id_>>kBTLCrystalOffset)&kBTLCrystalMask) + 1; }

  /** Returns BTL iphi index for crystal according to type tile or bar */
  int iphi( CrysLayout lay ) const ;

  /** Returns BTL ieta index for crystal according to type tile or bar */
  int ietaAbs( CrysLayout lay ) const ;

  int ieta( CrysLayout lay ) const { return zside()*ietaAbs( lay ); }

  /** define a dense index of arrays from a DetId */
  int hashedIndex( CrysLayout lay ) const ;

  static bool validHashedIndex( uint32_t din ) { return ( din < kSizeForDenseIndexing ) ; }

  /** get a DetId from a compact index for arrays */
  BTLDetId getUnhashedIndex( int hi, CrysLayout lay ) const ;

};

std::ostream& operator<< ( std::ostream&, const BTLDetId& );

#endif // DataFormats_BTLDetId_BTLDetId_h

