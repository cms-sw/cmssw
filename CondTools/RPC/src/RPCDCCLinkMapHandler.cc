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

RPCDCCLinkMapHandler::RPCDCCLinkMapHandler(edm::ParameterSet const & config)
    : id_(config.getParameter<std::string>("identifier"))
    , data_tag_(config.getParameter<std::string>("dataTag"))
    , since_run_(config.getParameter<unsigned long long>("sinceRun"))
    , txt_file_(config.getUntrackedParameter<std::string>("txtFile", ""))
    , connect_(config.getParameter<std::string>("connect"))
{
    edm::LogInfo("RPCDCCLinkMapHandler") << "Configuring Input Connection";
    connection_.setParameters(config.getParameter<edm::ParameterSet>("DBParameters"));
    connection_.configure();
}

RPCDCCLinkMapHandler::~RPCDCCLinkMapHandler()
{}

void RPCDCCLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCDCCLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & tag_info = tagInfo();
    if (since_run_ < tag_info.lastInterval.first)
        throw cms::Exception("RPCDCCLinkMapHandler") << "Refuse to create RPCDCCLinkMap for run " << since_run_
                                                     << ", older than most recent tag" << tag_info.lastInterval.first;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Opening read-only Input Session";
    auto input_session = connection_.createCoralSession(connect_, false); // writeCapable
    edm::LogInfo("RPCDCCLinkMapHandler") << "Started Input Transaction";
    input_session->transaction().start(true); // readOnly

    std::unique_ptr<coral::IQuery> query(input_session->schema("CMS_RPC_CONF").newQuery());
    query->addToTableList("DCCBOARD");
    query->addToTableList("TRIGGERBOARD");
    query->addToTableList("BOARDBOARDCONN");
    query->addToTableList("BOARD");

    query->addToOutputList("DCCBOARD.FEDNUMBER", "DCC");
    query->addToOutputList("TRIGGERBOARD.DCCINPUTCHANNELNUM", "DCC_INPUT");
    query->addToOutputList("BOARDBOARDCONN.COLLECTORBOARDINPUTNUM", "TB_INPUT");
    query->addToOutputList("BOARD.NAME", "LB_NAME");

    coral::AttributeList query_condition_data;
    query->setCondition("TRIGGERBOARD.DCCBOARD_DCCBOARDID=DCCBOARD.DCCBOARDID"
                        " AND BOARDBOARDCONN.BOARD_COLLECTORBOARDID=TRIGGERBOARD.TRIGGERBOARDID"
                        " AND BOARD.BOARDID=BOARDBOARDCONN.BOARD_BOARDID"
                        , query_condition_data);

    int dcc(0), dcc_input(0), tb_input(0);
    std::string lb_name("");

    std::unique_ptr<RPCDCCLinkMap> dcc_link_map_object(new RPCDCCLinkMap());
    RPCDCCLinkMap::map_type & dcc_link_map
        = dcc_link_map_object->getMap();
    RPCLBLink lb_link;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Execute query";
    coral::ICursor & cursor(query->execute());
    while (cursor.next()) {
        coral::AttributeList const & row(cursor.currentRow());

        // RPCLBLink
        lb_name = row["LB_NAME"].data<std::string>();
        RPCLBLinkNameParser::parse(lb_name, lb_link);
        if (lb_name != lb_link.getName())
            edm::LogWarning("RPCDCCLinkMapHandler") << "Mismatch LinkBoard Name: " << lb_name << " vs " << lb_link;
        lb_link.setLinkBoard().setConnector(); // MLB to link

        dcc = row["DCC"].data<long long>();
        dcc_input = row["DCC_INPUT"].data<long long>();
        tb_input = row["TB_INPUT"].data<long long>();

        dcc_link_map.insert(std::pair<RPCDCCLink, RPCLBLink>(RPCDCCLink(dcc, dcc_input, tb_input)
                                                             , lb_link));
    }
    cursor.close();

    input_session->transaction().commit();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCDCCLinkMapHandler") << "Fill txtFile";
        std::ofstream ofstream(txt_file_);
        for (auto const & link : dcc_link_map) {
            ofstream << link.first << ": " << link.second << std::endl;
        }
    }

    edm::LogInfo("RPCDCCLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(dcc_link_map_object.release(), since_run_));
}

std::string RPCDCCLinkMapHandler::id() const
{
    return id_;
}
