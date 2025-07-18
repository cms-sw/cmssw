#include "RecoTracker/LSTCore/interface/ModuleConnectionMap.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

lst::ModuleConnectionMap::ModuleConnectionMap() {}

lst::ModuleConnectionMap::ModuleConnectionMap(std::string const& filename) { load(filename); }

void lst::ModuleConnectionMap::load(std::string const& filename) {
  moduleConnections_.clear();

  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  while (!ifile.eof()) {
    unsigned int detid, number_of_connections;

    // Read the detid and the number of connections from the binary file
    ifile.read(reinterpret_cast<char*>(&detid), sizeof(detid));
    ifile.read(reinterpret_cast<char*>(&number_of_connections), sizeof(number_of_connections));

    if (ifile) {
      std::vector<unsigned int> connected_detids;
      connected_detids.reserve(number_of_connections);

      // Read the connections for the given detid
      for (unsigned int i = 0; i < number_of_connections; ++i) {
        unsigned int connected_detid;
        ifile.read(reinterpret_cast<char*>(&connected_detid), sizeof(connected_detid));
        if (ifile) {
          connected_detids.push_back(connected_detid);
        } else {
          if (!ifile.eof()) {
            throw std::runtime_error("Failed to read connection data.");
          }
          break;  // Exit loop on read failure that's not EOF
        }
      }

      if (ifile) {
        moduleConnections_[detid] = std::move(connected_detids);
      }
    } else {
      if (!ifile.eof()) {
        throw std::runtime_error("Failed to read module connection binary data.");
      }
    }
  }
}

void lst::ModuleConnectionMap::add(std::string const& filename) {
  std::ifstream ifile;
  ifile.open(filename.c_str());
  std::string line;

  while (std::getline(ifile, line)) {
    unsigned int detid;
    int number_of_connections;
    std::vector<unsigned int> connected_detids;
    unsigned int connected_detid;

    std::stringstream ss(line);

    ss >> detid >> number_of_connections;
    connected_detids.reserve(number_of_connections);

    for (int ii = 0; ii < number_of_connections; ++ii) {
      ss >> connected_detid;
      connected_detids.push_back(connected_detid);
    }

    auto& thisModuleConnections = moduleConnections_.at(detid);

    // Concatenate
    thisModuleConnections.insert(thisModuleConnections.end(), connected_detids.begin(), connected_detids.end());

    // Sort
    std::sort(thisModuleConnections.begin(), thisModuleConnections.end());

    // Unique
    thisModuleConnections.erase(std::unique(thisModuleConnections.begin(), thisModuleConnections.end()),
                                thisModuleConnections.end());
  }
}

void lst::ModuleConnectionMap::print() {
  std::cout << "Printing ModuleConnectionMap" << std::endl;
  for (auto& pair : moduleConnections_) {
    unsigned int detid = pair.first;
    std::vector<unsigned int> const& connected_detids = pair.second;
    std::cout << " detid: " << detid << std::endl;
    for (auto& connected_detid : connected_detids) {
      std::cout << " connected_detid: " << connected_detid << std::endl;
    }
  }
}

const std::vector<unsigned int>& lst::ModuleConnectionMap::getConnectedModuleDetIds(unsigned int detid) const {
  static const std::vector<unsigned int> dummy;
  auto const mList = moduleConnections_.find(detid);
  return mList != moduleConnections_.end() ? mList->second : dummy;
}
int lst::ModuleConnectionMap::size() const { return moduleConnections_.size(); }
