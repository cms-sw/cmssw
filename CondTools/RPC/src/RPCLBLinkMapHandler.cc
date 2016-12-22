#include "CondTools/RPC/interface/RPCLBLinkMapHandler.h"

#include <fstream>
#include <memory>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondCore/CondDB/interface/Types.h"

#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/IQueryDefinition.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"

#include "CondTools/RPC/interface/RPCLBLinkNameParser.h"

RPCDetId RPCLBLinkMapHandler::getRPCDetId(int _region, int _disk_or_wheel, int _layer, int _sector
                                          , std::string _subsector_string, std::string _partition)
{
    int _station(0), _ring(0), _subsector(0), _roll(0);
    // region well-defined
    if (!_region) { // barrel
        switch (_layer) {
        case 1:
        case 2: _station = 1; break;
        case 3:
        case 4: _station = 2; _layer -= 2; break;
        case 5: _station = 3; _layer = 1; break;
        case 6: _station = 4; _layer = 1; break;
        }
        _ring = _disk_or_wheel;
        // sector well-defined
        _subsector = 1;
        if (_subsector_string == "+"
            && (_station == 3
                || (_station==4 && (_sector != 4 && _sector != 9 && _sector != 11)))
            )
            _subsector = 2;
        if (_station == 4 && _sector == 4) {
            if (_subsector_string == "-")
                _subsector=2;
            else if (_subsector_string == "+")
                _subsector=3;
            else if (_subsector_string == "++")
                _subsector=4;
        }
    } else { // endcap
        _station = std::abs(_disk_or_wheel);
        _ring = _layer;
        _layer = 1;
        if (_ring > 1 || _station == 1) {
            _subsector = (_sector - 1) % 6 + 1;
            _sector = (_sector - 1) / 6 + 1;
        } else {
            _subsector = (_sector - 1) % 3 + 1;
            _sector = (_sector - 1) / 3 + 1;
        }
    }
    // roll
    if (_partition == "Backward" || _partition == "A")
        _roll = 1;
    else if (_partition == "Central" || _partition == "B")
        _roll = 2;
    else if (_partition == "Forward" || _partition == "C")
        _roll = 3;
    else if (_partition == "D")
        _roll = 4;
    else
        throw cms::Exception("RPCLBLinkMapHandler") << "Unexpected partition name " << _partition;

    return RPCDetId(_region, _ring, _station, _sector, _layer, _subsector, _roll);
}

RPCLBLinkMapHandler::RPCLBLinkMapHandler(edm::ParameterSet const & _config)
    : id_(_config.getParameter<std::string>("identifier"))
    , data_tag_(_config.getParameter<std::string>("dataTag"))
    , since_run_(_config.getParameter<unsigned long long>("sinceRun"))
    , txt_file_(_config.getUntrackedParameter<std::string>("txtFile", ""))
    , connect_(_config.getParameter<std::string>("connect"))
{
    edm::LogInfo("RPCDCCLinkMapHandler") << "Configuring Input Connection";
    connection_.setParameters(_config.getParameter<edm::ParameterSet>("DBParameters"));
    connection_.configure();
}

RPCLBLinkMapHandler::~RPCLBLinkMapHandler()
{}

void RPCLBLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCLBLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & _tag_info = tagInfo();
    if (since_run_ < _tag_info.lastInterval.first)
        throw cms::Exception("RPCLBLinkMapHandler") << "Refuse to create RPCLBLinkMap for run " << since_run_
                                                    << ", older than most recent tag" << _tag_info.lastInterval.first;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Opening read-only Input Session";
    auto _input_session = connection_.createCoralSession(connect_, false); // writeCapable
    edm::LogInfo("RPCDCCLinkMapHandler") << "Started Input Transaction";
    _input_session->transaction().start(true); // readOnly

    std::auto_ptr<coral::IQuery> _query(_input_session->schema("CMS_RPC_CONF").newQuery());
    _query->addToTableList("BOARD");
    _query->addToTableList("CHAMBERSTRIP");
    _query->addToTableList("CHAMBERLOCATION");
    _query->addToTableList("FEBLOCATION");
    _query->addToTableList("FEBCONNECTOR");
    coral::IQueryDefinition & _subquery_min_channel(_query->defineSubQuery("MIN_CHANNEL"));
    _query->addToTableList("MIN_CHANNEL");
    coral::IQueryDefinition & _subquery_max_strip(_query->defineSubQuery("MAX_STRIP"));
    _query->addToTableList("MAX_STRIP");

    _query->addToOutputList("BOARD.NAME", "LB_NAME");
    _query->addToOutputList("FEBCONNECTOR.LINKBOARDINPUTNUM", "CONNECTOR");
    _query->addToOutputList("CHAMBERSTRIP.CHAMBERSTRIPNUMBER", "FIRST_STRIP");
    _query->addToOutputList("CAST(DECODE(SIGN(MAX_STRIP.STRIP - CHAMBERSTRIP.CHAMBERSTRIPNUMBER), 1, 1, -1) AS INTEGER)", "SLOPE");
    _query->addToOutputList("MIN_CHANNEL.CHANNELS", "CHANNELS");
    _query->addToOutputList("CAST(DECODE(CHAMBERLOCATION.BARRELORENDCAP, 'Barrel', 0, DECODE(SIGN(CHAMBERLOCATION.DISKORWHEEL), 1, 1, -1)) AS INTEGER)", "REGION");
    _query->addToOutputList("CHAMBERLOCATION.DISKORWHEEL", "DISKORWHEEL");
    _query->addToOutputList("CHAMBERLOCATION.LAYER", "LAYER");
    _query->addToOutputList("CHAMBERLOCATION.SECTOR", "SECTOR");
    _query->addToOutputList("CHAMBERLOCATION.SUBSECTOR", "SUBSECTOR");
    _query->addToOutputList("FEBLOCATION.FEBLOCALETAPARTITION", "ETAPARTITION");

    _subquery_min_channel.addToTableList("CHAMBERSTRIP");
    _subquery_min_channel.addToOutputList("FC_FEBCONNECTORID");
    _subquery_min_channel.addToOutputList("MIN(CABLECHANNELNUM)", "CHANNEL");
    _subquery_min_channel.addToOutputList("CAST(SUM(POWER(2, CABLECHANNELNUM-1)) AS INTEGER)", "CHANNELS");
    _subquery_min_channel.groupBy("FC_FEBCONNECTORID");
    coral::AttributeList _subquery_min_channel_condition_data;
    _subquery_min_channel.setCondition("CABLECHANNELNUM IS NOT NULL"
                                       , _subquery_min_channel_condition_data);

    _subquery_max_strip.addToTableList("CHAMBERSTRIP");
    coral::IQueryDefinition & _subquery_max_channel(_subquery_max_strip.defineSubQuery("MAX_CHANNEL"));
    _subquery_max_strip.addToTableList("MAX_CHANNEL");
    _subquery_max_strip.addToOutputList("CHAMBERSTRIP.FC_FEBCONNECTORID", "FC_FEBCONNECTORID");
    _subquery_max_strip.addToOutputList("CHAMBERSTRIP.CHAMBERSTRIPNUMBER", "STRIP");
    coral::AttributeList _subquery_max_strip_condition_data;
    _subquery_max_strip.setCondition("CHAMBERSTRIP.FC_FEBCONNECTORID=MAX_CHANNEL.FC_FEBCONNECTORID"
                                     " AND CHAMBERSTRIP.CABLECHANNELNUM=MAX_CHANNEL.CHANNEL"
                                     , _subquery_max_strip_condition_data);

    _subquery_max_channel.addToTableList("CHAMBERSTRIP");
    _subquery_max_channel.addToOutputList("FC_FEBCONNECTORID");
    _subquery_max_channel.addToOutputList("MAX(CABLECHANNELNUM)", "CHANNEL");
    _subquery_max_channel.groupBy("FC_FEBCONNECTORID");
    coral::AttributeList _subquery_max_channel_condition_data;
    _subquery_max_channel.setCondition("CABLECHANNELNUM IS NOT NULL"
                                       , _subquery_max_channel_condition_data);

    coral::AttributeList _query_condition_data;
    _query->setCondition("CHAMBERSTRIP.FC_FEBCONNECTORID=MIN_CHANNEL.FC_FEBCONNECTORID"
                         " AND CHAMBERSTRIP.CABLECHANNELNUM=MIN_CHANNEL.CHANNEL"
                         " AND CHAMBERSTRIP.FC_FEBCONNECTORID=MAX_STRIP.FC_FEBCONNECTORID"
                         " AND CHAMBERSTRIP.FC_FEBCONNECTORID=FEBCONNECTOR.FEBCONNECTORID"
                         " AND FEBCONNECTOR.FL_FEBLOCATIONID=FEBLOCATION.FEBLOCATIONID"
                         " AND BOARD.BOARDID=FEBLOCATION.LB_LINKBOARDID"
                         " AND CHAMBERLOCATION.CHAMBERLOCATIONID=FEBLOCATION.CL_CHAMBERLOCATIONID"
                         , _query_condition_data);

    std::string _lb_name("");
    int _first_strip(0), _slope(1);
    ::uint16_t _channels(0x0);

    RPCLBLinkMap * _lb_link_map_object = new RPCLBLinkMap();
    RPCLBLinkMap::map_type & _lb_link_map
        = _lb_link_map_object->getMap();
    RPCLBLink _lb_link;
    RPCDetId _det_id;
    std::string _subsector;

    edm::LogInfo("RPCLBLinkMapHandler") << "Execute query";
    coral::ICursor & _cursor(_query->execute());
    while (_cursor.next()) {
        coral::AttributeList const & _row(_cursor.currentRow());

        // RPCLBLink
        _lb_name = _row["LB_NAME"].data<std::string>();
        RPCLBLinkNameParser::parse(_lb_name, _lb_link);
        if (_lb_name != _lb_link.getName())
            edm::LogWarning("RPCLBLinkMapHandler") << "Mismatch LinkBoard Name: " << _lb_name << " vs " << _lb_link;
        _lb_link.setConnector(_row["CONNECTOR"].data<short>() - 1); // 1-6 to 0-5

        // RPCDetId
        if (_row["SUBSECTOR"].isNull())
            _subsector = "";
        else
            _subsector = _row["SUBSECTOR"].data<std::string>();
        _det_id = getRPCDetId(_row["REGION"].data<long long>()
                              , _row["DISKORWHEEL"].data<short>()
                              , _row["LAYER"].data<short>()
                              , _row["SECTOR"].data<short>()
                              , _subsector
                              , _row["ETAPARTITION"].data<std::string>());

        // RPCFebConnector
        _first_strip = _row["FIRST_STRIP"].data<int>();
        _slope = _row["SLOPE"].data<long long>();
        _channels = (::uint16_t)(_row["CHANNELS"].data<long long>());

        _lb_link_map.insert(std::pair<RPCLBLink, RPCFebConnector>(_lb_link, RPCFebConnector(_det_id
                                                                                            , _first_strip
                                                                                            , _slope
                                                                                            , _channels)));
    }
    _cursor.close();

    _input_session->transaction().commit();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCLBLinkMapHandler") << "Fill txtFile";
        std::ofstream _ofstream(txt_file_);
        for (RPCLBLinkMap::map_type::const_iterator _link_connector = _lb_link_map.begin()
                 ; _link_connector != _lb_link_map.end() ; ++_link_connector) {
            _ofstream << _link_connector->first << ": " << _link_connector->second << std::endl;
        }
    }

    edm::LogInfo("RPCLBLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(_lb_link_map_object, since_run_));
}

std::string RPCLBLinkMapHandler::id() const
{
    return id_;
}
