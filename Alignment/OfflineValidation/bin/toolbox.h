#ifndef _TOOLBOX_
#define _TOOLBOX_
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

namespace AllInOneConfig {

  inline void dump(const boost::property_tree::ptree& tree) {
    for (const auto& it : tree) {
      auto key = it.first, value = tree.get<std::string>(key);
      std::cout << key << '\t' << value << '\n';
    }
    std::cout << std::flush;
  }

}  // namespace AllInOneConfig
#endif
