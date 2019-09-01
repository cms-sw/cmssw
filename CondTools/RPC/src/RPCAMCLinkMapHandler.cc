#include "CondTools/RPC/interface/RPCAMCLinkMapHandler.h"

#include <fstream>
#include <memory>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCAMCLinkMapHandler::RPCAMCLinkMapHandler(edm::ParameterSet const& config)
    : id_(config.getParameter<std::string>("identifier")),
      data_tag_(config.getParameter<std::string>("dataTag")),
      since_run_(config.getParameter<unsigned long long>("sinceRun")),
      input_file_(config.getParameter<edm::FileInPath>("inputFile").fullPath()),
      wheel_not_side_(config.getParameter<bool>("wheelNotSide")),
      wos_fed_(config.getParameter<std::vector<int> >("wheelOrSideFED")),
      n_sectors_(config.getParameter<unsigned int>("nSectors")),
      wos_sector_amc_(wos_fed_.size(), std::vector<int>(n_sectors_, 0)),
      txt_file_(config.getUntrackedParameter<std::string>("txtFile", "")) {
  std::vector<long long> wos_sector_amc_packed(config.getParameter<std::vector<long long> >("wheelOrSideSectorAMC"));

  if (wos_fed_.size() != wos_sector_amc_packed.size()) {
    throw cms::Exception("RPCAMCLinkMapHandler") << "Refuse to handle inconsistent input: "
                                                 << "sizes of wheelOrSideFED and wheelOrSideSectorAMC don't match";
  }

  std::vector<std::vector<int> >::iterator sector_amc = wos_sector_amc_.begin();
  for (std::vector<long long>::const_iterator sector_amc_packed = wos_sector_amc_packed.begin();
       sector_amc_packed != wos_sector_amc_packed.end();
       ++sector_amc_packed, ++sector_amc) {
    for (unsigned int sector = 0; sector < n_sectors_; ++sector) {
      sector_amc->at(sector) = ((*sector_amc_packed) >> (4 * (n_sectors_ - sector - 1))) & 0xf;
    }
  }
}

RPCAMCLinkMapHandler::~RPCAMCLinkMapHandler() {}

void RPCAMCLinkMapHandler::getNewObjects() {
  edm::LogInfo("RPCAMCLinkMapHandler") << "getNewObjects";
  cond::TagInfo const& tag_info = tagInfo();
  if (since_run_ < tag_info.lastInterval.first) {
    throw cms::Exception("RPCAMCLinkMapHandler") << "Refuse to create RPCAMCLinkMap for run " << since_run_
                                                 << ", older than most recent tag" << tag_info.lastInterval.first;
  }

  std::string amc_name, link_name;
  unsigned int wos, sector, amc_number, amc_input;

  std::unique_ptr<RPCAMCLinkMap> amc_link_map_object(new RPCAMCLinkMap());
  RPCAMCLinkMap::map_type& amc_link_map = amc_link_map_object->getMap();
  RPCLBLink lb_link;

  std::string line;
  std::istringstream conv;

  std::ifstream input_file(input_file_);

  input_file >> amc_name >> amc_input >> link_name;
  std::getline(input_file, line);

  while (input_file) {
    // parse AMC Slot Name - no checking: failure is an error
    if (wheel_not_side_) {                     // wheel
      std::string::size_type pos(2), next(2);  // skip YB
      int wheel;
      next = amc_name.find_first_not_of("+-0123456789", pos);
      conv.clear();
      conv.str(amc_name.substr(pos, next - pos));
      conv >> wheel;
      wos = wheel + 2;

      pos = next + 2;
      next = amc_name.find_first_not_of("+-0123456789", pos);
      if (next == std::string::npos)
        next = amc_name.size();
      conv.clear();
      conv.str(amc_name.substr(pos, next - pos));
      conv >> sector;
    } else {                                  // side
      wos = (amc_name.at(4) == 'n' ? 0 : 1);  // skip CPPF or OMTF
      conv.clear();
      conv.str(amc_name.substr(5, 1));
      conv >> sector;
    }

    if (sector > n_sectors_) {
      throw cms::Exception("RPCAMCLinkMapHandler")
          << "Found sector greater than the number of sectors: " << sector << " > " << n_sectors_;
    }

    if (wos >= wos_fed_.size()) {
      throw cms::Exception("RPCAMCLinkMapHandler")
          << "Found " << (wheel_not_side_ ? "wheel" : "side") << " outside range: " << wos << " >= " << wos_fed_.size();
    }

    amc_number = wos_sector_amc_.at(wos).at(sector - 1);

    RPCLBLinkNameParser::parse(link_name, lb_link);

    amc_link_map.insert(std::pair<RPCAMCLink, RPCLBLink>(RPCAMCLink(wos_fed_.at(wos), amc_number, amc_input), lb_link));

    input_file >> amc_name >> amc_input >> link_name;
    std::getline(input_file, line);
  }
  input_file.close();

  if (!txt_file_.empty()) {
    edm::LogInfo("RPCAMCLinkMapHandler") << "Fill txtFile";
    std::ofstream ofstream(txt_file_);
    for (auto const link : amc_link_map) {
      ofstream << link.first << ": " << link.second << std::endl;
    }
  }

  edm::LogInfo("RPCAMCLinkMapHandler") << "Add to transfer list";
  m_to_transfer.push_back(std::make_pair(amc_link_map_object.release(), since_run_));
}

std::string RPCAMCLinkMapHandler::id() const { return id_; }
