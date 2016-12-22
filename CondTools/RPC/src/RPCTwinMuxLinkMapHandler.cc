#include "CondTools/RPC/interface/RPCTwinMuxLinkMapHandler.h"

#include <fstream>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCTwinMuxLinkMapHandler::RPCTwinMuxLinkMapHandler(edm::ParameterSet const & _config)
    : id_(_config.getParameter<std::string>("identifier"))
    , data_tag_(_config.getParameter<std::string>("dataTag"))
    , since_run_(_config.getParameter<unsigned long long>("sinceRun"))
    , input_file_(_config.getParameter<edm::FileInPath>("inputFile").fullPath())
    , wheel_fed_(_config.getParameter<std::vector<int> >("wheelFED"))
    , wheel_sector_amc_(std::vector<std::vector<int> >(5, std::vector<int>(12, 0)))
    , txt_file_(_config.getUntrackedParameter<std::string>("txtFile", ""))
{
    std::vector<long long> _wheel_sector_amc_packed(_config.getParameter<std::vector<long long> >("wheelSectorAMC"));
    std::vector<std::vector<int> >::iterator _sector_amc = wheel_sector_amc_.begin();
    for (std::vector<long long>::const_iterator _sector_amc_packed = _wheel_sector_amc_packed.begin()
             ; _sector_amc_packed != _wheel_sector_amc_packed.end() ; ++_sector_amc_packed, ++_sector_amc)
        for (unsigned int _sector = 0 ; _sector < 12 ; ++_sector)
            _sector_amc->at(_sector) = ((*_sector_amc_packed) >> (11*4-4*_sector)) & 0xf;
}

RPCTwinMuxLinkMapHandler::~RPCTwinMuxLinkMapHandler()
{}

void RPCTwinMuxLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCTwinMuxLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & _tag_info = tagInfo();
    if (since_run_ < _tag_info.lastInterval.first)
        throw cms::Exception("RPCTwinMuxLinkMapHandler") << "Refuse to create RPCTwinMuxLinkMap for run " << since_run_
                                                         << ", older than most recent tag" << _tag_info.lastInterval.first;

    std::string _tm_name, _link_name;
    int _wheel, _fed, _sector, _amc_number, _tm_input;

    RPCAMCLinkMap * _twinmux_link_map_object = new RPCAMCLinkMap();
    RPCAMCLinkMap::map_type & _twinmux_link_map
        = _twinmux_link_map_object->getMap();
    RPCLBLink _lb_link;

    std::string _line;
    std::istringstream _conv;

    std::ifstream _input_file(input_file_);

    _input_file >> _tm_name >> _tm_name >> _tm_input >> _link_name;
    std::getline(_input_file, _line);

    while (_input_file) {
        // parse AMC Slot Name - no checking: failure is an error
        std::string::size_type _pos(2), _next(2);
        _next = _tm_name.find_first_not_of("+-0123456789", _pos);
        _conv.clear();
        _conv.str(_tm_name.substr(_pos, _next - _pos));
        _conv >> _wheel;

        _pos = _next + 2;
        _next = _tm_name.find_first_not_of("+-0123456789", _pos);
        if (_next == std::string::npos)
            _next = _tm_name.size();
        _conv.clear();
        _conv.str(_tm_name.substr(_pos, _next - _pos));
        _conv >> _sector;

        _fed = wheel_fed_.at(_wheel + 2);
        _amc_number = wheel_sector_amc_.at(_wheel + 2).at(_sector - 1);

        RPCLBLinkNameParser::parse(_link_name, _lb_link);

        _twinmux_link_map.insert(std::pair<RPCAMCLink, RPCLBLink>(RPCAMCLink(_fed, _amc_number, _tm_input)
                                                                  , _lb_link));

        _input_file >> _tm_name >> _tm_name >> _tm_input >> _link_name;
        std::getline(_input_file, _line);
    }
    _input_file.close();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCTwinMuxLinkMapHandler") << "Fill txtFile";
        std::ofstream _ofstream(txt_file_);
        for (RPCAMCLinkMap::map_type::const_iterator _link = _twinmux_link_map.begin()
                 ; _link != _twinmux_link_map.end() ; ++_link) {
            _ofstream << _link->first << ": " << _link->second << std::endl;
        }
    }

    edm::LogInfo("RPCTwinMuxLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(_twinmux_link_map_object, since_run_));
}

std::string RPCTwinMuxLinkMapHandler::id() const
{
    return id_;
}
