#include "CondTools/RPC/interface/RPCCPPFLinkMapHandler.h"

#include <fstream>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCCPPFLinkMapHandler::RPCCPPFLinkMapHandler(edm::ParameterSet const & _config)
    : id_(_config.getParameter<std::string>("identifier"))
    , data_tag_(_config.getParameter<std::string>("dataTag"))
    , since_run_(_config.getParameter<unsigned long long>("sinceRun"))
    , input_file_(_config.getParameter<edm::FileInPath>("inputFile").fullPath())
    , side_fed_(_config.getParameter<std::vector<int> >("sideFED"))
    , n_sectors_(_config.getParameter<unsigned int>("nSectors"))
    , side_sector_amc_(std::vector<std::vector<int> >(2, std::vector<int>(n_sectors_, 0)))
    , txt_file_(_config.getUntrackedParameter<std::string>("txtFile", ""))
{
    std::vector<long long> _side_sector_amc_packed(_config.getParameter<std::vector<long long> >("sideSectorAMC"));
    std::vector<std::vector<int> >::iterator _sector_amc = side_sector_amc_.begin();
    for (std::vector<long long>::const_iterator _sector_amc_packed = _side_sector_amc_packed.begin()
             ; _sector_amc_packed != _side_sector_amc_packed.end() ; ++_sector_amc_packed, ++_sector_amc) {
        for (unsigned int _sector = 0 ; _sector < n_sectors_ ; ++_sector) {
            _sector_amc->at(_sector) = ((*_sector_amc_packed) >> (4 * (n_sectors_ - _sector - 1))) & 0xf;
        }
    }
}

RPCCPPFLinkMapHandler::~RPCCPPFLinkMapHandler()
{}

void RPCCPPFLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCCPPFLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & _tag_info = tagInfo();
    if (since_run_ < _tag_info.lastInterval.first)
        throw cms::Exception("RPCCPPFLinkMapHandler") << "Refuse to create RPCCPPFLinkMap for run " << since_run_
                                                      << ", older than most recent tag" << _tag_info.lastInterval.first;

    std::string _cppf_name, _link_name;
    int _side, _sector, _amc_number, _cppf_input;

    RPCAMCLinkMap * _cppf_link_map_object = new RPCAMCLinkMap();
    RPCAMCLinkMap::map_type & _cppf_link_map
        = _cppf_link_map_object->getMap();
    RPCLBLink _lb_link;

    std::string _line;
    std::istringstream _conv;

    std::ifstream _input_file(input_file_);

    _input_file >> _cppf_name >> _cppf_input >> _link_name;
    std::getline(_input_file, _line);

    while (_input_file) {
        // parse AMC Slot Name - no checking: failure is an error
        _side = (_cppf_name.at(4) == 'n' ? 0 : 1);
        _conv.clear();
        _conv.str(_cppf_name.substr(5, 1));
        _conv >> _sector;

        _amc_number = side_sector_amc_.at(_side).at(_sector - 1);

        RPCLBLinkNameParser::parse(_link_name, _lb_link);

        _cppf_link_map.insert(std::pair<RPCAMCLink, RPCLBLink>(RPCAMCLink(side_fed_.at(_side), _amc_number, _cppf_input)
                                                               , _lb_link));

        _input_file >> _cppf_name >> _cppf_input >> _link_name;
        std::getline(_input_file, _line);
    }
    _input_file.close();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCCPPFLinkMapHandler") << "Fill txtFile";
        std::ofstream _ofstream(txt_file_);
        for (auto const _link : _cppf_link_map) {
            _ofstream << _link.first << ": " << _link.second << std::endl;
        }
    }

    edm::LogInfo("RPCCPPFLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(_cppf_link_map_object, since_run_));
}

std::string RPCCPPFLinkMapHandler::id() const
{
    return id_;
}
