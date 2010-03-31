#include "DataFormats/Math/interface/SSEVec.h"
#include <ostream>
std::ostream & operator<<(std::ostream & out,  mathSSE::Vec3F const & v) {
  return out << '(' << v.arr[0] <<", " << v.arr[1] <<", "<< v.arr[2] <<')';
}
