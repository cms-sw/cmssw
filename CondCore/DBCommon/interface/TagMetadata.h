#ifndef DBCommon_TagMetadata_h
#define DBCommon_TagMetadata_h
#include <string>
#include <boost/functional/hash.hpp>
namespace cond{
  class TagMetadata{
  public:
    std::string tag;
    std::string pfn;
    std::string recordname;
    std::string labelname;
    std::string objectname;
    std::size_t hashvalue()const{
      boost::hash<std::string> hasher;
      std::size_t result=hasher(tag+pfn);
      return result;
    }
    bool operator<(const TagMetadata& toCompare ) const {
      return this->hashvalue()<toCompare.hashvalue();
    }
  };
}
#endif
