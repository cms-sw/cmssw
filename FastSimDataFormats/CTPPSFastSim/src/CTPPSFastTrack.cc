#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrack.h"
#include <ostream>
//float t,float xi,unsigned int cellid ,float tof,const Vector &momentum,const Point &vertex)
std::ostream& operator<<(std::ostream& o, const CTPPSFastTrack& hit) {
  return o << hit.t() << " " << hit.xi() << " " << hit.cellid() << "  " << hit.tof();
}
