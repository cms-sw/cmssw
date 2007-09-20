#ifndef COND_TAGDBNAMES_H
#define COND_TAGDBNAMES_H
#include <string>
namespace cond{
  class TagDBNames {
  public:
    TagDBNames(){}
    static const std::string& tagTreeTable();
    static const std::string& tagInventoryTable();
  };
}
#endif
