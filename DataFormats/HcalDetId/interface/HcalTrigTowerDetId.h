#ifndef DATAFORMATS_HCALDETID_HCALTRIGTOWERDETID_H
#define DATAFORMATS_HCALDETID_HCALTRIGTOWERDETID_H 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class HcalTrigTowerDetId
    
Cell id for an Calo Trigger tower

   \author J. Mans - Minnesota
*/
class HcalTrigTowerDetId : public DetId {
public:
  static const int kHcalPhiMask       = 0x7F;
  static const int kHcalEtaOffset     = 7;
  static const int kHcalEtaMask       = 0x3F;
  static const int kHcalZsideMask     = 0x2000;
  static const int kHcalDepthOffset   = 14;
  static const int kHcalDepthMask     = 0x7;
  static const int kHcalVersOffset    = 17;
  static const int kHcalVersMask      = 0x7;
 public:
   /** Constructor of a null id */
  HcalTrigTowerDetId();
  /** Constructor from a raw value */
  HcalTrigTowerDetId(uint32_t rawid);  
  /** \brief Constructor from signed ieta, iphi
  */
  HcalTrigTowerDetId(int ieta, int iphi);
  /** \brief Constructor from signed ieta, iphi, depth
  */
  HcalTrigTowerDetId(int ieta, int iphi, int depth);
  /** \brief Constructor from signed ieta, iphi, depth, version
  */
  HcalTrigTowerDetId(int ieta, int iphi, int depth, int version);

  /** Constructor from a generic cell id */
  HcalTrigTowerDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HcalTrigTowerDetId& operator=(const DetId& id);

  void setVersion(int version);

  /// get the subdetector
  HcalSubdetector subdet() const { return (HcalSubdetector)(subdetId()); }
  /// get the z-side of the tower (1/-1)
  int zside() const { return (id_&kHcalZsideMask)?(1):(-1); }
  /// get the absolute value of the tower ieta
  int ietaAbs() const { return (id_>>kHcalEtaOffset)&kHcalEtaMask; }
  /// get the tower ieta
  int ieta() const { return zside()*ietaAbs(); }
  /// get the tower iphi
  int iphi() const { return id_&kHcalPhiMask; }
  /// get the depth (zero for LHC Run 1, may be nonzero for later runs)
  int depth() const { return (id_>>kHcalDepthOffset)&kHcalDepthMask; }
  /// get the version code for the trigger tower
  int version() const { return (id_>>kHcalVersOffset)&kHcalVersMask; }

  static const HcalTrigTowerDetId Undefined;

};

std::ostream& operator<<(std::ostream&,const HcalTrigTowerDetId& id);


#endif
