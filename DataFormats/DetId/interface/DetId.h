#ifndef DATAFORMATS_DETID_H
#define DATAFORMATS_DETID_H 1

#include <boost/cstdint.hpp>
#include <ostream>

namespace cms {

/** \class DetId
    
   $Date: 2005/07/19 18:21:57 $
   $Revision: 1.1 $
*/
class DetId {
public:
  enum Detector { Tracker=1,Muon=2,Ecal=3,Hcal=4 };
  /// Create an empty or null id (also for persitence)
  DetId();
  /// Create an id from a raw number
  DetId(uint32_t id);
  /// Create an id, filling the detector and subdetector fields as specified
  DetId(Detector det, int subdet);

  /// get the detector field from this detid
  Detector det() const { return Detector((id_>>28)&0xF); }
  /// get the contents of the subdetector field (should be protected?)
  int subdetId() const { return ((id_>>25)&0x7); }

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

}

//std::ostream& operator<<(std::ostream& s, const DetId& id);

#endif
