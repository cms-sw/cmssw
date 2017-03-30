#include "CondTools/RPC/interface/RPCCPPFLinkMapHandler.h"

#include <fstream>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCCPPFLinkMapHandler::RPCCPPFLinkMapHandler(edm::ParameterSet const & config)
    : id_(config.getParameter<std::string>("identifier"))
    , data_tag_(config.getParameter<std::string>("dataTag"))
    , since_run_(config.getParameter<unsigned long long>("sinceRun"))
    , input_file_(config.getParameter<edm::FileInPath>("inputFile").fullPath())
    , side_fed_(config.getParameter<std::vector<int> >("sideFED"))
    , n_sectors_(config.getParameter<unsigned int>("nSectors"))
    , side_sector_amc_(std::vector<std::vector<int> >(2, std::vector<int>(n_sectors_, 0)))
    , txt_file_(config.getUntrackedParameter<std::string>("txtFile", ""))
{
    std::vector<long long> side_sector_amc_packed(config.getParameter<std::vector<long long> >("sideSectorAMC"));
    std::vector<std::vector<int> >::iterator sector_amc = side_sector_amc_.begin();
    for (std::vector<long long>::const_iterator sector_amc_packed = side_sector_amc_packed.begin()
             ; sector_amc_packed != side_sector_amc_packed.end() ; ++sector_amc_packed, ++sector_amc) {
        for (unsigned int sector = 0 ; sector < n_sectors_ ; ++sector) {
            sector_amc->at(sector) = ((*sector_amc_packed) >> (4 * (n_sectors_ - sector - 1))) & 0xf;
        }
    }
}

RPCCPPFLinkMapHandler::~RPCCPPFLinkMapHandler()
{}

void RPCCPPFLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCCPPFLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & tag_info = tagInfo();
    if (since_run_ < tag_info.lastInterval.first)
        throw cms::Exception("RPCCPPFLinkMapHandler") << "Refuse to create RPCCPPFLinkMap for run " << since_run_
                                                      << ", older than most recent tag" << tag_info.lastInterval.first;

    std::string cppf_name, link_name;
    int side, sector, amc_number, cppf_input;

    RPCAMCLinkMap * cppf_link_map_object = new RPCAMCLinkMap();
    RPCAMCLinkMap::map_type & cppf_link_map
        = cppf_link_map_object->getMap();
    RPCLBLink lb_link;

    std::string line;
    std::istringstream conv;

    std::ifstream input_file(input_file_);

    input_file >> cppf_name >> cppf_input >> link_name;
    std::getline(input_file, line);

    while (input_file) {
        // parse AMC Slot Name - no checking: failure is an error
        side = (cppf_name.at(4) == 'n' ? 0 : 1);
        conv.clear();
        conv.str(cppf_name.substr(5, 1));
        conv >> sector;

        amc_number = side_sector_amc_.at(side).at(sector - 1);

        RPCLBLinkNameParser::parse(link_name, lb_link);

        cppf_link_map.insert(std::pair<RPCAMCLink, RPCLBLink>(RPCAMCLink(side_fed_.at(side), amc_number, cppf_input)
                                                              , lb_link));

        input_file >> cppf_name >> cppf_input >> link_name;
        std::getline(input_file, line);
    }
    input_file.close();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCCPPFLinkMapHandler") << "Fill txtFile";
        std::ofstream ofstream(txt_file_);
        for (auto const link : cppf_link_map) {
            ofstream << link.first << ": " << link.second << std::endl;
        }
    }

    edm::LogInfo("RPCCPPFLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(cppf_link_map_object, since_run_));
}

std::string RPCCPPFLinkMapHandler::id() const
{
    return id_;
}
