#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <algorithm>
#include <iterator>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Need a mapping file as argument!" << std::endl;
    return -1;
  }

  //build entity list map
  hgcal::mappingtools::HGCalEntityList map;
  edm::FileInPath fip(argv[1]);
  std::string url(fip.fullPath());
  map.buildFrom(url);

  //print out for debugging purposes
  auto cols = map.getColumnNames();
  auto entities = map.getEntries();
  std::cout << "Read " << entities.size() << " entities from " << url << std::endl << "Columns available are {";
  std::copy(cols.begin(), cols.end(), std::ostream_iterator<std::string>(std::cout, ","));
  std::cout << "}" << std::endl;

  //line-by-line
  for (auto r : entities) {
    for (auto c : cols)
      std::cout << map.getAttr(c, r) << " ";
    std::cout << std::endl;
  }

  return 0;
}
