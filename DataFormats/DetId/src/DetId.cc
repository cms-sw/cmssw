#include "DataFormats/DetId/interface/DetId.h"

namespace cms {

DetId::DetId() : id_(0) { }
DetId::DetId(uint32_t id) : id_(id) { }
DetId::DetId(Detector det, int subdet) {
  id_=((det&0xF)<<28)|((subdet&0x7)<<25);
}
 

}
