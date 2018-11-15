#ifndef DATAFORMATS_HCALDETID_HCALDETID_H
#define DATAFORMATS_HCALDETID_HCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"


/** \class HcalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells only
 */
class HcalDetId : public DetId {

public:
  static constexpr uint32_t kHcalPhiMask1       = 0x7F;
  static constexpr uint32_t kHcalPhiMask2       = 0x3FF;
  static constexpr uint32_t kHcalEtaOffset1     = 7;
  static constexpr uint32_t kHcalEtaOffset2     = 10;
  static constexpr uint32_t kHcalEtaMask1       = 0x3F;
  static constexpr uint32_t kHcalEtaMask2       = 0x1FF;
  static constexpr uint32_t kHcalZsideMask1     = 0x2000;
  static constexpr uint32_t kHcalZsideMask2     = 0x80000;
  static constexpr uint32_t kHcalDepthOffset1   = 14;
  static constexpr uint32_t kHcalDepthOffset2   = 20;
  static constexpr uint32_t kHcalDepthMask1     = 0x1F;
  static constexpr uint32_t kHcalDepthMask2     = 0xF;
  static constexpr uint32_t kHcalDepthSet1      = 0x1C000;
  static constexpr uint32_t kHcalDepthSet2      = 0xF00000;
  static constexpr uint32_t kHcalIdFormat2      = 0x1000000;
  static constexpr uint32_t kHcalIdMask         = 0xFE000000;

public:
  /** Create a null cellid*/
  constexpr HcalDetId() : DetId() {
  }
  /** Create cellid from raw id (0=invalid tower id) */
  constexpr HcalDetId(uint32_t rawid) {
    if ((DetId::Detector(rawid>>DetId::kDetOffset)&DetId::kDetMask) != Hcal) {
      id_ = rawid;
    } else  {
      HcalSubdetector subdet = (HcalSubdetector)((rawid>>DetId::kSubdetOffset)&DetId::kSubdetMask);
      if ((subdet==HcalBarrel) || (subdet==HcalEndcap) ||
	  (subdet==HcalOuter) || (subdet==HcalForward)) {
        id_ = newForm(rawid);
      } else {
        id_ = rawid;
      }
    }
  }

  /** Constructor from subdetector, signed tower ieta,iphi,and depth */
  constexpr HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
    // (no checking at this point!)
    id_ |= (kHcalIdFormat2) | ((depth&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((tower_ieta>0)?(kHcalZsideMask2|(tower_ieta<<kHcalEtaOffset2)):((-tower_ieta)<<kHcalEtaOffset2)) |
      (tower_iphi&kHcalPhiMask2);
  }
  /** Constructor from a generic cell id */
  constexpr HcalDetId(const DetId& gen) {
    if (!gen.null()) {
      HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
      if (gen.det()!=Hcal ||
	  (subdet!=HcalBarrel && subdet!=HcalEndcap &&
	   subdet!=HcalOuter && subdet!=HcalForward &&
	   subdet!=HcalTriggerTower && subdet!=HcalOther)) {
        throw cms::Exception("Invalid DetId") << "Cannot initialize HcalDetId from " << std::hex << gen.rawId() << std::dec;
      }
      if ((subdet==HcalBarrel) || (subdet==HcalEndcap) ||
	  (subdet==HcalOuter) || (subdet==HcalForward)) {
        id_ = newForm(gen.rawId());
      } else {
        id_ = gen.rawId();
      }
    } else {
      id_ = gen.rawId();
    }
  }
  /** Assignment from a generic cell id */
  constexpr HcalDetId& operator=(const DetId& gen) {
    if (!gen.null()) {
      HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
      if (gen.det()!=Hcal ||
      (subdet!=HcalBarrel && subdet!=HcalEndcap &&
       subdet!=HcalOuter && subdet!=HcalForward &&
       subdet!=HcalTriggerTower && subdet!=HcalOther)) {
        throw cms::Exception("Invalid DetId") << "Cannot assign HcalDetId from " << std::hex << gen.rawId() << std::dec;
      }
      if ((subdet==HcalBarrel) || (subdet==HcalEndcap) ||
      (subdet==HcalOuter) || (subdet==HcalForward)) {
        id_ = newForm(gen.rawId());
      } else {
        id_ = gen.rawId();
      }
    } else {
      id_ = gen.rawId();
    }
    return (*this);
  }

  /** Comparison operator */
  constexpr bool operator==(DetId gen) const {
    uint32_t rawid = gen.rawId();
    if (rawid == id_) return true;
    int zsid{0}, eta{0}, phi{0}, dep{0};
    unpackId(rawid, zsid, eta, phi, dep);
    bool result = (((id_&kHcalIdMask) == (rawid&kHcalIdMask)) && 
		   (zsid==zside()) && (eta==ietaAbs()) && (phi==iphi()) &&
		   (dep==depth()));
    return result;
  }

  constexpr bool operator!=(DetId gen) const {
    uint32_t rawid = gen.rawId();
    if (rawid == id_) return false;
    int zsid{0}, eta{0}, phi{0}, dep{0};
    unpackId(rawid, zsid, eta, phi, dep);
    bool result = (((id_&kHcalIdMask)!=(rawid&kHcalIdMask)) ||
		   (zsid!=zside()) || (eta!=ietaAbs()) || 
		   (phi!=iphi()) || (dep!=depth()));
    return result;
  }

  constexpr bool operator<(DetId gen) const {
    uint32_t rawid = gen.rawId();
    if ((rawid&kHcalIdFormat2)==(id_&kHcalIdFormat2)) {
      return id_<rawid;
    } else {
      int zsid{0}, eta{0}, phi{0}, dep{0};
      unpackId(rawid, zsid, eta, phi, dep);
      rawid &= kHcalIdMask;
      if (oldFormat()) {
        rawid |= (((dep&kHcalDepthMask1)<<kHcalDepthOffset1) |
		  ((zsid>0)?(kHcalZsideMask1|(eta<<kHcalEtaOffset1)):((eta)<<kHcalEtaOffset1)) |
		  (phi&kHcalPhiMask1));
      } else {
        rawid |= ((kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
		  ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
		  (phi&kHcalPhiMask2));
      }
      return (id_<rawid);
    }
  }

  /// get the subdetector
  constexpr HcalSubdetector subdet() const { return (HcalSubdetector)(subdetId()); }
  constexpr bool oldFormat() const { return ((id_&kHcalIdFormat2)==0)?(true):(false); }
  /// get the z-side of the cell (1/-1)
  constexpr int zside() const {
    if (oldFormat()) return (id_&kHcalZsideMask1)?(1):(-1);
    else             return (id_&kHcalZsideMask2)?(1):(-1);
  }
  /// get the absolute value of the cell ieta
  constexpr int ietaAbs() const {
    if (oldFormat()) return (id_>>kHcalEtaOffset1)&kHcalEtaMask1;
    else             return (id_>>kHcalEtaOffset2)&kHcalEtaMask2;
  }
  /// get the cell ieta
  constexpr int ieta() const { return zside()*ietaAbs(); }
  /// get the cell iphi
  constexpr int iphi() const {
    if (oldFormat()) return id_&kHcalPhiMask1;
    else             return id_&kHcalPhiMask2;
  }
  /// get the tower depth
  constexpr int depth() const {
    if (oldFormat())  return (id_>>kHcalDepthOffset1)&kHcalDepthMask1;
    else              return (id_>>kHcalDepthOffset2)&kHcalDepthMask2;
  }
  /// get full depth information for HF
  constexpr int hfdepth() const {
    int dep = depth();
    if (subdet() == HcalForward) {
      if (dep > 2) dep -= 2;
    }
    return dep;
  }
  /// get the tower depth
  constexpr uint32_t maskDepth() const {
    if (oldFormat())  return (id_|kHcalDepthSet1);
    else              return (id_|kHcalDepthSet2);
  }
  /// change format
  constexpr uint32_t otherForm() const {
    uint32_t rawid = (id_&kHcalIdMask);
    if (oldFormat()) {
      rawid = newForm(id_);
    } else {
      rawid |= ((depth()&kHcalDepthMask1)<<kHcalDepthOffset1) |
        ((ieta()>0)?(kHcalZsideMask1|(ieta()<<kHcalEtaOffset1)):((-ieta())<<kHcalEtaOffset1)) |
        (iphi()&kHcalPhiMask1);
    }
    return rawid;
  }
  constexpr void changeForm() {
    id_ = otherForm();
  }
  constexpr uint32_t newForm() const {
    return newForm(id_);
  }
  constexpr static int32_t newForm(const uint32_t& inpid) {
    uint32_t rawid(inpid);
    if ((rawid&kHcalIdFormat2)==0) {
      int zsid{0}, eta{0}, phi{0}, dep{0};
      unpackId(rawid, zsid, eta, phi, dep);
      rawid    = inpid&kHcalIdMask;
      rawid   |= ((kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
		  ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
		  (phi&kHcalPhiMask2));
    }
    return rawid;
  }
  /// base detId for HF dual channels
  constexpr bool sameBaseDetId(const DetId& gen) const {
    uint32_t rawid = gen.rawId();
    if (rawid == id_) return true;
    int zsid{0}, eta{0}, phi{0}, dep{0};
    if ((id_&kHcalIdMask) != (rawid&kHcalIdMask)) return false;
    unpackId(rawid, zsid, eta, phi, dep);
    if (subdet() == HcalForward && dep > 2) dep -= 2;
    bool result = ((zsid==zside()) && (eta==ietaAbs()) && (phi==iphi()) &&
		   (dep==hfdepth()));
    return result;
  }
  constexpr HcalDetId baseDetId() const {
    if (subdet() != HcalForward || depth() <= 2) {
      return HcalDetId(id_);
    } else {
      int zsid{0}, eta{0}, phi{0}, dep{0};
      unpackId(id_, zsid, eta, phi, dep);
      dep     -= 2;
      uint32_t rawid    = id_&kHcalIdMask;
      rawid   |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
        ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
        (phi&kHcalPhiMask2);
      return HcalDetId(rawid);
    }
  }
  /// second PMT anode detId for HF dual channels
  constexpr HcalDetId secondAnodeId() const {
    if (subdet() != HcalForward || depth() > 2) {
      return HcalDetId(id_);
    } else {
      int zsid{0}, eta{0}, phi{0}, dep{0};
      unpackId(id_, zsid, eta, phi, dep);
      dep     += 2;
      uint32_t rawid    = id_&kHcalIdMask;
      rawid   |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
        ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
        (phi&kHcalPhiMask2);
      return HcalDetId(rawid);
    }
  }

  /// get the smallest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  constexpr int crystal_ieta_low() const { return ((ieta()-zside())*5)+zside(); }
  /// get the largest crystal_ieta of the crystal in front of this tower (HB and HE tower 17 only)
  constexpr int crystal_ieta_high() const { return ((ieta()-zside())*5)+5*zside(); }
  /// get the smallest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  constexpr int crystal_iphi_low() const {
    int simple_iphi=((iphi()-1)*5)+1;
    simple_iphi+=10;
    return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
  }
  /// get the largest crystal_iphi of the crystal in front of this tower (HB and HE tower 17 only)
  constexpr int crystal_iphi_high() const {
    int simple_iphi=((iphi()-1)*5)+5;
    simple_iphi+=10;
    return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
  }

  static const HcalDetId Undefined;

private:

  constexpr void newFromOld(const uint32_t& rawid) {
   id_ = newForm(rawid);
  }

  constexpr static void unpackId(const uint32_t& rawid, int& zsid, int& eta, int& phi,
       int& dep) {
    if ((rawid&kHcalIdFormat2)==0) {
      zsid = (rawid&kHcalZsideMask1)?(1):(-1);
      eta  = (rawid>>kHcalEtaOffset1)&kHcalEtaMask1;
      phi  = rawid&kHcalPhiMask1;
      dep  = (rawid>>kHcalDepthOffset1)&kHcalDepthMask1;
    } else {
      zsid = (rawid&kHcalZsideMask2)?(1):(-1);
      eta  = (rawid>>kHcalEtaOffset2)&kHcalEtaMask2;
      phi  = rawid&kHcalPhiMask2;
      dep  = (rawid>>kHcalDepthOffset2)&kHcalDepthMask2;
    }
  }
};

std::ostream& operator<<(std::ostream&,const HcalDetId& id);

#endif
