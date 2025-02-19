#ifndef HcalZDCDetId_h_included
#define HcalZDCDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class HcalZDCDetId
  *  
  *  Contents of the HcalZDCDetId :
  *     [6]   Z position (true for positive)
  *     [5:4] Section (EM/HAD/Lumi)
  *     [3:0] Channel
  *
  * $Date: 2009/02/09 16:48:01 $
  * $Revision: 1.5 $
  * \author J. Mans - Minnesota
  */
class HcalZDCDetId : public DetId {
public:
  enum Section { Unknown=0, EM=1, HAD=2, LUM=3 };
  // 1 => CaloTower, 3 => Castor
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
  int zside() const { return (id_&0x40)?(1):(-1); }
  /// get the section
  Section section() const { return (Section)((id_>>4)&0x3); }
  /// get the depth (1 for EM, channel + 1 for HAD, not sure yet for LUM, leave as default)
  int depth() const { return (((id_>>4)&0x3)==1)?(1):((((id_>>4)&0x3)==2)?((id_&0xF)+1):(id_&0xF)); }
  /// get the channel 
  int channel() const { return id_&0xF; }

  uint32_t denseIndex() const ;

  static bool validDenseIndex( uint32_t di ) { return ( di < kSizeForDenseIndexing ) ; }

  static HcalZDCDetId detIdFromDenseIndex( uint32_t di ) ;

  static bool validDetId( Section se, int dp ) ;

   private:

      enum { kDepEM  = 5,
	     kDepHAD = 4,
	     kDepLUM = 2,
	     kDepTot = kDepEM + kDepHAD + kDepLUM };

   public:

      enum { kSizeForDenseIndexing = 2*kDepTot } ;

};

std::ostream& operator<<(std::ostream&,const HcalZDCDetId& id);


#endif // HcalZDCDetId_h_included
