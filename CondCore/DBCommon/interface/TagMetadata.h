#ifndef DBCommon_TagMetadata_h
#define DBCommon_TagMetadata_h
#include <string>
namespace cond{
  class TagMetadata{
  public:
    std::string pfn;
    std::string recordname;
    std::string objectname;
    std::string labelname;
  };
}
#endif
