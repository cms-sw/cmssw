#ifndef _TOOLBOX_
#define _TOOLBOX_
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

namespace AllInOneConfig {

  //boost::property_tree::ptree get_unique_child (const boost::property_tree::ptree& tree)
  //{
  //    // checking that there is only one child
  //    int i = 0;
  //    for (const auto& it: tree) ++i;
  //
  //    if (i > 1) {
  //        std::cerr << "A single job expects only one geometry, but " << i << " were found.\n";
  //        exit(EXIT_FAILURE);
  //    }
  //
  //    // returning the child
  //    for (const auto& it: tree)
  //        return it.second;
  //
  //    std::cerr << "No validation found.\n";
  //    exit(EXIT_FAILURE);
  //}

  void dump(const boost::property_tree::ptree& tree) {
    for (const auto& it : tree) {
      auto key = it.first, value = tree.get<std::string>(key);
      std::cout << key << '\t' << value << '\n';
    }
    std::cout << std::flush;
  }

}  // namespace AllInOneConfig
#endif
