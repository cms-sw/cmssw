#include "headers.h"

namespace CondFormats_MFObjects {
  struct dictionary {
    std::pair<std::string, int> str_int;
    std::pair<int, std::pair<std::string, int> > tablemappair;
    std::pair<const int, std::pair<std::string, int> > tablemappairc;
    std::map<int, std::pair<std::string, int> > tablemap;
    MagFieldConfig conf;
  };
}
