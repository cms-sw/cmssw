#ifndef DATAFORMATS_DETID_H
#define DATAFORMATS_DETID_H 1

#include <boost/cstdint.hpp>
#include <ostream>

/** \class DetId

Parent class for all detector ids in CMS.  The DetId is a 32-bit
unsigned integer.  The four most significant bits ([31:28]) identify
the large-scale detector (e.g. Tracker or Ecal) while the next three
bits ([27:25]) identify a part of the detector (such as HcalBarrel
(HB) for Hcal).

$Date: 2005/10/06 00:23:46 $
$Revision: 1.4 $
*/
class DetId {
public:
  static const int kDetOffset          = 28;
  static const int kSubdetOffset       = 25;


  enum Detector { Tracker=1,Muon=2,Ecal=3,Hcal=4,Calo=5 };
  /// Create an empty or null id (also for persistence)
  DetId();
  /// Create an id from a raw number
  explicit DetId(uint32_t id);
  /// Create an id, filling the detector and subdetector fields as specified
  DetId(Detector det, int subdet);

  /// get the detector field from this detid
  Detector det() const { return Detector((id_>>kDetOffset)&0xF); }
  /// get the contents of the subdetector field (not cast into any detector's numbering enum)
  int subdetId() const { return ((id_>>kSubdetOffset)&0x7); }

  uint32_t operator()() { return id_; }

  /// get the raw id 
  uint32_t rawId() const { return id_; }
  /// is this a null id ?
  bool null() const { return id_==0; }
  
  /// equality
  int operator==(const DetId& id) const { return id_==id.id_; }
  /// inequality
  int operator!=(const DetId& id) const { return id_!=id.id_; }
  /// comparison
  int operator<(const DetId& id) const { return id_<id.id_; }

protected:
  uint32_t id_;
};

//std::ostream& operator<<(std::ostream& s, const DetId& id);

#endif
