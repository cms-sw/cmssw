
#ifndef L1Trigger_DemonstratorTools_FileFormat_h
#define L1Trigger_DemonstratorTools_FileFormat_h

#include <iosfwd>

namespace l1t::demo {

  enum class FileFormat { APx, EMP, X20 };

  std::ostream& operator<<(std::ostream&, FileFormat);

}  // namespace l1t::demo

#endif
