/** \class RPCBookFolderStructure
 *
 * \author Anna Cimmino (INFN Napoli)
 *
 * Create folder structure for DQM histo saving
 */
#ifndef RPCBookFolderStructure_H
#define RPCBookFolderStructure_H

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <string>
#include <fmt/format.h>

struct RPCBookFolderStructure {
  static std::string folderStructure(const RPCDetId& detId) {
    if (detId.region() == 0)
      return fmt::format("Barrel/Wheel_{}/sector_{}/station_{}", detId.ring(), detId.sector(), detId.station());
    else if (detId.region() == -1)
      return fmt::format("Endcap-/Disk_-{}/ring_{}/sector_{}", detId.station(), detId.ring(), detId.sector());
    else if (detId.region() == 1)
      return fmt::format("Endcap+/Disk_{}/ring_{}/sector_{}", detId.station(), detId.ring(), detId.sector());
    return "Error/Folder/Creation";
  }
};

#endif
