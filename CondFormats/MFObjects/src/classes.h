#include "headers.h"

namespace CondFormats_MFObjects {
  struct dictionary {
    std::pair<std::string, int> str_int;
    std::map<unsigned int, std::pair<std::string, int> > tablemap;
    MagFieldConfig conf;
  };
}
