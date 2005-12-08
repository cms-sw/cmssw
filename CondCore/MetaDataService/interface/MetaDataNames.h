#ifndef COND_METADATANAMES_H
#define COND_METADATANAMES_H
#include <string>
namespace cond{
  class MetaDataNames {
  public:
    MetaDataNames(){}
    static const std::string& metadataTable();
    static const std::string& tagColumn();
    static const std::string& tokenColumn();
  };
}
#endif
