#include <cstdlib>
#include <iostream>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include "boost/filesystem.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

int single(int argc, char* argv[]) {
  // parse the command line
  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);

  pt::ptree alignment = main_tree.get_child("alignment");
  pt::ptree validation = main_tree.get_child("validation");

  // just dump the content, to show that it can be used
  dump(alignment);
  dump(validation);

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<single>(argc, argv); }
#endif
