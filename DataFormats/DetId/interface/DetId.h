#ifndef DATAFORMATS_DETID_H
#define DATAFORMATS_DETID_H


//FIXME shall be removed and implemented where the operator is defined
#include <ostream>

#include <cstdint>
/** \class DetId

Parent class for all detector ids in CMS.  The DetId is a 32-bit
unsigned integer.  The four most significant bits ([31:28]) identify
the large-scale detector (e.g. Tracker or Ecal) while the next three
bits ([27:25]) identify a part of the detector (such as HcalBarrel
(HB) for Hcal).

*/
class DetId {
public:
  static const int kDetMask            = 0xF;
  static const int kSubdetMask         = 0x7;
  static const int kDetOffset          = 28;
  static const int kSubdetOffset       = 25;


  enum Detector {Tracker=1, Muon=2, Ecal=3, Hcal=4, Calo=5, Forward=6,
		 VeryForward=7, HGCalEE=8, HGCalHSi=9, HGCalHSc=10,
		 HGCalTrigger=11};
  /// Create an empty or null id (also for persistence)
  constexpr DetId()  : id_(0) { }
  /// Create an id from a raw number
  constexpr DetId(uint32_t id) : id_(id) { }
  /// Create an id, filling the detector and subdetector fields as specified
  constexpr DetId(Detector det, int subdet) 
    : id_(((det&kDetMask)<<kDetOffset)|((subdet&kSubdetMask)<<kSubdetOffset))
  {}

  /// get the detector field from this detid
  constexpr Detector det() const { return Detector((id_>>kDetOffset)&kDetMask); }
  /// get the contents of the subdetector field (not cast into any detector's numbering enum)
  constexpr int subdetId() const { 
    return ((HGCalEE==det()) || (HGCalHSi==det()) || (HGCalHSc==det()) ?
	    0 : ((id_>>kSubdetOffset)&kSubdetMask));
  }

  constexpr uint32_t operator()() const { return id_; }
  constexpr operator uint32_t() const { return id_; }

  /// get the raw id 
  constexpr uint32_t rawId() const { return id_; }
  /// is this a null id ?
  constexpr bool null() const { return id_==0; }
  
  /// equality
  constexpr bool operator==(DetId id) const { return id_==id.id_; }
  /// inequality
  constexpr bool operator!=(DetId id) const { return id_!=id.id_; }
  /// comparison
  constexpr bool operator<(DetId id) const { return id_<id.id_; }

protected:
  uint32_t id_;
};

/// equality
constexpr inline bool operator==(uint32_t i, DetId id)  { return i==id(); }
constexpr inline bool operator==(DetId id, uint32_t i)  { return i==id(); }
/// inequality
constexpr inline bool operator!=(uint32_t i, DetId id)  { return i!=id(); }
constexpr inline bool operator!=(DetId id, uint32_t i) { return i!=id(); }
/// comparison
constexpr inline bool operator<(uint32_t i, DetId id) { return i<id(); }
constexpr inline bool operator<(DetId id, uint32_t i) { return id()<i; }


//std::ostream& operator<<(std::ostream& s, const DetId& id);

namespace std {
  template<> struct hash<DetId> {
    typedef DetId argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& id) const noexcept {
      return std::hash<uint32_t>()(id.rawId());            
    }
  };
}

#endif
