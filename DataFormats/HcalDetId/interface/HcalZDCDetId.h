#ifndef DataFormats_HcalDetId_HcalZDCDetId_h_included
#define DataFormats_HcalDetId_HcalZDCDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class HcalZDCDetId
  *  
  *  Contents of the HcalZDCDetId :
  *     [7]   Set for RPD
  *     [6]   Z position (true for positive)
  *     [5:4] Section (EM/HAD/Lumi)
  *     [3:0] Channel
  *
  */
class HcalZDCDetId : public DetId {
public:
  static const int kZDCChannelMask   = 0xF;
  static const int kZDCSectionMask   = 0x3;
  static const int kZDCSectionOffset = 4;
  static const int kZDCZsideMask     = 0x40;
  static const int kZDCRPDMask       = 0x80;
  enum Section {Unknown=0, EM=1, HAD=2, LUM=3, RPD=4};

  static const int SubdetectorId = 2;

  /** Create a null cellid*/
  HcalZDCDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HcalZDCDetId(uint32_t rawid);
  /** Constructor from section, eta sign, and channel */
  HcalZDCDetId(Section section, bool true_for_positive_eta, int channel);
  /** Constructor from a generic cell id */
  HcalZDCDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalZDCDetId& operator=(const DetId& id);

  /// get the z-side of the cell (1/-1)
  int zside() const { return ((id_&kZDCZsideMask) ? (1) : (-1)); }
  /// get the section
  Section section() const;
  /// get the depth (1 for EM, channel + 1 for HAD, 2 for RPD, not sure yet for LUM, leave as default)
  int depth() const; 
  /// get the channel 
  int channel() const;

  uint32_t denseIndex() const ;

  static bool validDenseIndex( uint32_t di ) { return ( di < kSizeForDenseIndexing ) ; }

  static HcalZDCDetId detIdFromDenseIndex( uint32_t di ) ;

  static bool validDetId( Section se, int dp ) ;

private:

  enum { kDepEM  = 5,
	 kDepHAD = 4,
	 kDepLUM = 2,
	 kDepRPD = 16,
	 kDepRun1= kDepEM + kDepHAD + kDepLUM,
	 kDepTot = kDepRun1 + kDepRPD};

public:

  enum { kSizeForDenseIndexing = 2*kDepRun1 } ;

};

std::ostream& operator<<(std::ostream&,const HcalZDCDetId& id);

#endif // DataFormats_HcalDetId_HcalZDCDetId_h_included
