#include "CondTools/RPC/interface/RPCDCCLinkMapHandler.h"

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

RPCDCCLinkMapHandler::RPCDCCLinkMapHandler(edm::ParameterSet const & _config)
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

RPCDCCLinkMapHandler::~RPCDCCLinkMapHandler()
{}

void RPCDCCLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCDCCLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & _tag_info = tagInfo();
    if (since_run_ < _tag_info.lastInterval.first)
        throw cms::Exception("RPCDCCLinkMapHandler") << "Refuse to create RPCDCCLinkMap for run " << since_run_
                                                     << ", older than most recent tag" << _tag_info.lastInterval.first;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Opening read-only Input Session";
    auto _input_session = connection_.createCoralSession(connect_, false); // writeCapable
    edm::LogInfo("RPCDCCLinkMapHandler") << "Started Input Transaction";
    _input_session->transaction().start(true); // readOnly

    std::auto_ptr<coral::IQuery> _query(_input_session->schema("CMS_RPC_CONF").newQuery());
    _query->addToTableList("DCCBOARD");
    _query->addToTableList("TRIGGERBOARD");
    _query->addToTableList("BOARDBOARDCONN");
    _query->addToTableList("BOARD");

    _query->addToOutputList("DCCBOARD.FEDNUMBER", "DCC");
    _query->addToOutputList("TRIGGERBOARD.DCCINPUTCHANNELNUM", "DCC_INPUT");
    _query->addToOutputList("BOARDBOARDCONN.COLLECTORBOARDINPUTNUM", "TB_INPUT");
    _query->addToOutputList("BOARD.NAME", "LB_NAME");

    coral::AttributeList _query_condition_data;
    _query->setCondition("TRIGGERBOARD.DCCBOARD_DCCBOARDID=DCCBOARD.DCCBOARDID"
                         " AND BOARDBOARDCONN.BOARD_COLLECTORBOARDID=TRIGGERBOARD.TRIGGERBOARDID"
                         " AND BOARD.BOARDID=BOARDBOARDCONN.BOARD_BOARDID"
                         , _query_condition_data);

    int _dcc(0), _dcc_input(0), _tb_input(0);
    std::string _lb_name("");

    RPCDCCLinkMap * _dcc_link_map_object = new RPCDCCLinkMap();
    RPCDCCLinkMap::map_type & _dcc_link_map
        = _dcc_link_map_object->getMap();
    RPCLBLink _lb_link;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Execute query";
    coral::ICursor & _cursor(_query->execute());
    while (_cursor.next()) {
        coral::AttributeList const & _row(_cursor.currentRow());

        // RPCLBLink
        _lb_name = _row["LB_NAME"].data<std::string>();
        RPCLBLinkNameParser::parse(_lb_name, _lb_link);
        if (_lb_name != _lb_link.getName())
            edm::LogWarning("RPCDCCLinkMapHandler") << "Mismatch LinkBoard Name: " << _lb_name << " vs " << _lb_link;
        _lb_link.setLinkBoard().setConnector(); // MLB to link

        _dcc = _row["DCC"].data<long long>();
        _dcc_input = _row["DCC_INPUT"].data<long long>();
        _tb_input = _row["TB_INPUT"].data<long long>();

        _dcc_link_map.insert(std::pair<RPCDCCLink, RPCLBLink>(RPCDCCLink(_dcc, _dcc_input, _tb_input)
                                                              , _lb_link));
    }
    _cursor.close();

    _input_session->transaction().commit();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCDCCLinkMapHandler") << "Fill txtFile";
        std::ofstream _ofstream(txt_file_);
        for (auto const & _link : _dcc_link_map) {
            _ofstream << _link.first << ": " << _link.second << std::endl;
        }
    }

    edm::LogInfo("RPCDCCLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(_dcc_link_map_object, since_run_));
}

std::string RPCDCCLinkMapHandler::id() const
{
    return id_;
}
