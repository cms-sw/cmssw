#include "CondTools/RPC/interface/RPCTwinMuxLinkMapHandler.h"

#include <fstream>
#include <memory>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCTwinMuxLinkMapHandler::RPCTwinMuxLinkMapHandler(edm::ParameterSet const & config)
    : id_(config.getParameter<std::string>("identifier"))
    , data_tag_(config.getParameter<std::string>("dataTag"))
    , since_run_(config.getParameter<unsigned long long>("sinceRun"))
    , input_file_(config.getParameter<edm::FileInPath>("inputFile").fullPath())
    , wheel_fed_(config.getParameter<std::vector<int> >("wheelFED"))
    , wheel_sector_amc_(std::vector<std::vector<int> >(5, std::vector<int>(12, 0)))
    , txt_file_(config.getUntrackedParameter<std::string>("txtFile", ""))
{
    std::vector<long long> wheel_sector_amc_packed(config.getParameter<std::vector<long long> >("wheelSectorAMC"));
    std::vector<std::vector<int> >::iterator sector_amc = wheel_sector_amc_.begin();
    for (std::vector<long long>::const_iterator sector_amc_packed = wheel_sector_amc_packed.begin()
             ; sector_amc_packed != wheel_sector_amc_packed.end() ; ++sector_amc_packed, ++sector_amc)
        for (unsigned int sector = 0 ; sector < 12 ; ++sector)
            sector_amc->at(sector) = ((*sector_amc_packed) >> (11*4-4*sector)) & 0xf;
}

RPCTwinMuxLinkMapHandler::~RPCTwinMuxLinkMapHandler()
{}

void RPCTwinMuxLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCTwinMuxLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & tag_info = tagInfo();
    if (since_run_ < tag_info.lastInterval.first)
        throw cms::Exception("RPCTwinMuxLinkMapHandler") << "Refuse to create RPCTwinMuxLinkMap for run " << since_run_
                                                         << ", older than most recent tag" << tag_info.lastInterval.first;

    std::string tm_name, link_name;
    int wheel, fed, sector, amc_number, tm_input;

    std::unique_ptr<RPCAMCLinkMap> twinmux_link_map_object(new RPCAMCLinkMap());
    RPCAMCLinkMap::map_type & twinmux_link_map
        = twinmux_link_map_object->getMap();
    RPCLBLink lb_link;

    std::string line;
    std::istringstream conv;

    std::ifstream input_file(input_file_);

    input_file >> tm_name >> tm_name >> tm_input >> link_name;
    std::getline(input_file, line);

    while (input_file) {
        // parse AMC Slot Name - no checking: failure is an error
        std::string::size_type pos(2), next(2);
        next = tm_name.find_first_not_of("+-0123456789", pos);
        conv.clear();
        conv.str(tm_name.substr(pos, next - pos));
        conv >> wheel;

        pos = next + 2;
        next = tm_name.find_first_not_of("+-0123456789", pos);
        if (next == std::string::npos)
            next = tm_name.size();
        conv.clear();
        conv.str(tm_name.substr(pos, next - pos));
        conv >> sector;

        fed = wheel_fed_.at(wheel + 2);
        amc_number = wheel_sector_amc_.at(wheel + 2).at(sector - 1);

        RPCLBLinkNameParser::parse(link_name, lb_link);

        twinmux_link_map.insert(std::pair<RPCAMCLink, RPCLBLink>(RPCAMCLink(fed, amc_number, tm_input)
                                                                 , lb_link));

        input_file >> tm_name >> tm_name >> tm_input >> link_name;
        std::getline(input_file, line);
    }
    input_file.close();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCTwinMuxLinkMapHandler") << "Fill txtFile";
        std::ofstream ofstream(txt_file_);
        for (auto const & link : twinmux_link_map) {
            ofstream << link.first << ": " << link.second << std::endl;
        }
    }

    edm::LogInfo("RPCTwinMuxLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(twinmux_link_map_object.release(), since_run_));
}

std::string RPCTwinMuxLinkMapHandler::id() const
{
    return id_;
}
