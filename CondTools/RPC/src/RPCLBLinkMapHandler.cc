#include "CondTools/RPC/interface/RPCLBLinkMapHandler.h"

#include <cstdint>
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

RPCDetId RPCLBLinkMapHandler::getRPCDetId(int region, int disk_or_wheel, int layer, int sector
                                          , std::string subsector_string, std::string partition)
{
    int station(0), ring(0), subsector(0), roll(0);
    // region well-defined
    if (!region) { // barrel
        switch (layer) {
        case 1:
        case 2: station = 1; break;
        case 3:
        case 4: station = 2; layer -= 2; break;
        case 5: station = 3; layer = 1; break;
        case 6: station = 4; layer = 1; break;
        }
        ring = disk_or_wheel;
        // sector well-defined
        subsector = 1;
        if (subsector_string == "+"
            && (station == 3
                || (station==4 && (sector != 4 && sector != 9 && sector != 11)))
            )
            subsector = 2;
        if (station == 4 && sector == 4) {
            if (subsector_string == "-")
                subsector=2;
            else if (subsector_string == "+")
                subsector=3;
            else if (subsector_string == "++")
                subsector=4;
        }
    } else { // endcap
        station = std::abs(disk_or_wheel);
        ring = layer;
        layer = 1;
        if (ring > 1 || station == 1) {
            subsector = (sector - 1) % 6 + 1;
            sector = (sector - 1) / 6 + 1;
        } else {
            subsector = (sector - 1) % 3 + 1;
            sector = (sector - 1) / 3 + 1;
        }
    }
    // roll
    if (partition == "Backward" || partition == "A")
        roll = 1;
    else if (partition == "Central" || partition == "B")
        roll = 2;
    else if (partition == "Forward" || partition == "C")
        roll = 3;
    else if (partition == "D")
        roll = 4;
    else
        throw cms::Exception("RPCLBLinkMapHandler") << "Unexpected partition name " << partition;

    return RPCDetId(region, ring, station, sector, layer, subsector, roll);
}

RPCLBLinkMapHandler::RPCLBLinkMapHandler(edm::ParameterSet const & config)
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

RPCLBLinkMapHandler::~RPCLBLinkMapHandler()
{}

void RPCLBLinkMapHandler::getNewObjects()
{
    edm::LogInfo("RPCLBLinkMapHandler") << "getNewObjects";
    cond::TagInfo const & tag_info = tagInfo();
    if (since_run_ < tag_info.lastInterval.first)
        throw cms::Exception("RPCLBLinkMapHandler") << "Refuse to create RPCLBLinkMap for run " << since_run_
                                                    << ", older than most recent tag" << tag_info.lastInterval.first;

    edm::LogInfo("RPCDCCLinkMapHandler") << "Opening read-only Input Session";
    auto input_session = connection_.createCoralSession(connect_, false); // writeCapable
    edm::LogInfo("RPCDCCLinkMapHandler") << "Started Input Transaction";
    input_session->transaction().start(true); // readOnly

    std::unique_ptr<coral::IQuery> query(input_session->schema("CMS_RPC_CONF").newQuery());
    query->addToTableList("BOARD");
    query->addToTableList("CHAMBERSTRIP");
    query->addToTableList("CHAMBERLOCATION");
    query->addToTableList("FEBLOCATION");
    query->addToTableList("FEBCONNECTOR");
    coral::IQueryDefinition & subquery_min_channel(query->defineSubQuery("MIN_CHANNEL"));
    query->addToTableList("MIN_CHANNEL");
    coral::IQueryDefinition & subquery_max_strip(query->defineSubQuery("MAX_STRIP"));
    query->addToTableList("MAX_STRIP");

    query->addToOutputList("BOARD.NAME", "LB_NAME");
    query->addToOutputList("FEBCONNECTOR.LINKBOARDINPUTNUM", "CONNECTOR");
    query->addToOutputList("CHAMBERSTRIP.CHAMBERSTRIPNUMBER", "FIRST_STRIP");
    query->addToOutputList("CAST(DECODE(SIGN(MAX_STRIP.STRIP - CHAMBERSTRIP.CHAMBERSTRIPNUMBER), 1, 1, -1) AS INTEGER)", "SLOPE");
    query->addToOutputList("MIN_CHANNEL.CHANNELS", "CHANNELS");
    query->addToOutputList("CAST(DECODE(CHAMBERLOCATION.BARRELORENDCAP, 'Barrel', 0, DECODE(SIGN(CHAMBERLOCATION.DISKORWHEEL), 1, 1, -1)) AS INTEGER)", "REGION");
    query->addToOutputList("CHAMBERLOCATION.DISKORWHEEL", "DISKORWHEEL");
    query->addToOutputList("CHAMBERLOCATION.LAYER", "LAYER");
    query->addToOutputList("CHAMBERLOCATION.SECTOR", "SECTOR");
    query->addToOutputList("CHAMBERLOCATION.SUBSECTOR", "SUBSECTOR");
    query->addToOutputList("FEBLOCATION.FEBLOCALETAPARTITION", "ETAPARTITION");

    subquery_min_channel.addToTableList("CHAMBERSTRIP");
    subquery_min_channel.addToOutputList("FC_FEBCONNECTORID");
    subquery_min_channel.addToOutputList("MIN(CABLECHANNELNUM)", "CHANNEL");
    subquery_min_channel.addToOutputList("CAST(SUM(POWER(2, CABLECHANNELNUM-1)) AS INTEGER)", "CHANNELS");
    subquery_min_channel.groupBy("FC_FEBCONNECTORID");
    coral::AttributeList subquery_min_channel_condition_data;
    subquery_min_channel.setCondition("CABLECHANNELNUM IS NOT NULL"
                                      , subquery_min_channel_condition_data);

    subquery_max_strip.addToTableList("CHAMBERSTRIP");
    coral::IQueryDefinition & subquery_max_channel(subquery_max_strip.defineSubQuery("MAX_CHANNEL"));
    subquery_max_strip.addToTableList("MAX_CHANNEL");
    subquery_max_strip.addToOutputList("CHAMBERSTRIP.FC_FEBCONNECTORID", "FC_FEBCONNECTORID");
    subquery_max_strip.addToOutputList("CHAMBERSTRIP.CHAMBERSTRIPNUMBER", "STRIP");
    coral::AttributeList subquery_max_strip_condition_data;
    subquery_max_strip.setCondition("CHAMBERSTRIP.FC_FEBCONNECTORID=MAX_CHANNEL.FC_FEBCONNECTORID"
                                    " AND CHAMBERSTRIP.CABLECHANNELNUM=MAX_CHANNEL.CHANNEL"
                                    , subquery_max_strip_condition_data);

    subquery_max_channel.addToTableList("CHAMBERSTRIP");
    subquery_max_channel.addToOutputList("FC_FEBCONNECTORID");
    subquery_max_channel.addToOutputList("MAX(CABLECHANNELNUM)", "CHANNEL");
    subquery_max_channel.groupBy("FC_FEBCONNECTORID");
    coral::AttributeList subquery_max_channel_condition_data;
    subquery_max_channel.setCondition("CABLECHANNELNUM IS NOT NULL"
                                      , subquery_max_channel_condition_data);

    coral::AttributeList query_condition_data;
    query->setCondition("CHAMBERSTRIP.FC_FEBCONNECTORID=MIN_CHANNEL.FC_FEBCONNECTORID"
                        " AND CHAMBERSTRIP.CABLECHANNELNUM=MIN_CHANNEL.CHANNEL"
                        " AND CHAMBERSTRIP.FC_FEBCONNECTORID=MAX_STRIP.FC_FEBCONNECTORID"
                        " AND CHAMBERSTRIP.FC_FEBCONNECTORID=FEBCONNECTOR.FEBCONNECTORID"
                        " AND FEBCONNECTOR.FL_FEBLOCATIONID=FEBLOCATION.FEBLOCATIONID"
                        " AND BOARD.BOARDID=FEBLOCATION.LB_LINKBOARDID"
                        " AND CHAMBERLOCATION.CHAMBERLOCATIONID=FEBLOCATION.CL_CHAMBERLOCATIONID"
                        , query_condition_data);

    std::string lb_name("");
    int first_strip(0), slope(1);
    std::uint16_t channels(0x0);

    std::unique_ptr<RPCLBLinkMap> lb_link_map_object(new RPCLBLinkMap());
    RPCLBLinkMap::map_type & lb_link_map
        = lb_link_map_object->getMap();
    RPCLBLink lb_link;
    RPCDetId det_id;
    std::string subsector;

    edm::LogInfo("RPCLBLinkMapHandler") << "Execute query";
    coral::ICursor & cursor(query->execute());
    while (cursor.next()) {
        coral::AttributeList const & row(cursor.currentRow());

        // RPCLBLink
        lb_name = row["LB_NAME"].data<std::string>();
        RPCLBLinkNameParser::parse(lb_name, lb_link);
        if (lb_name != lb_link.getName())
            edm::LogWarning("RPCLBLinkMapHandler") << "Mismatch LinkBoard Name: " << lb_name << " vs " << lb_link;
        lb_link.setConnector(row["CONNECTOR"].data<short>() - 1); // 1-6 to 0-5

        // RPCDetId
        if (row["SUBSECTOR"].isNull())
            subsector = "";
        else
            subsector = row["SUBSECTOR"].data<std::string>();
        det_id = getRPCDetId(row["REGION"].data<long long>()
                             , row["DISKORWHEEL"].data<short>()
                             , row["LAYER"].data<short>()
                             , row["SECTOR"].data<short>()
                             , subsector
                             , row["ETAPARTITION"].data<std::string>());

        // RPCFebConnector
        first_strip = row["FIRST_STRIP"].data<int>();
        slope = row["SLOPE"].data<long long>();
        channels = (std::uint16_t)(row["CHANNELS"].data<long long>());

        lb_link_map.insert(std::pair<RPCLBLink, RPCFebConnector>(lb_link, RPCFebConnector(det_id
                                                                                          , first_strip
                                                                                          , slope
                                                                                          , channels)));
    }
    cursor.close();

    input_session->transaction().commit();

    if (!txt_file_.empty()) {
        edm::LogInfo("RPCLBLinkMapHandler") << "Fill txtFile";
        std::ofstream ofstream(txt_file_);
        for (auto const & link_connector : lb_link_map) {
            ofstream << link_connector.first << ": " << link_connector.second << std::endl;
        }
    }

    edm::LogInfo("RPCLBLinkMapHandler") << "Add to transfer list";
    m_to_transfer.push_back(std::make_pair(lb_link_map_object.release(), since_run_));
}

std::string RPCLBLinkMapHandler::id() const
{
    return id_;
}
