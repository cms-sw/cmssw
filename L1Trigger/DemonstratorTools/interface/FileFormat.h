
#ifndef L1Trigger_DemonstratorTools_FileFormat_h
#define L1Trigger_DemonstratorTools_FileFormat_h

#include <iosfwd>

namespace l1t::demo {

  enum class FileFormat {
    APx,
    EMPv1,  //< Format used in EMP until version 0.6.x
    EMPv2,  //< Format used in EMP from verison 0.7.0
    X2O
  };

  std::ostream& operator<<(std::ostream&, FileFormat);

}  // namespace l1t::demo

#endif
