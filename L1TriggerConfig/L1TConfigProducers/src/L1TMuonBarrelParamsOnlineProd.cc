#include <iostream>
#include <fstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TMuonBarrelParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonBarrelParamsO2ORcd,L1TMuonBarrelParams> {
private:
public:
    virtual boost::shared_ptr<L1TMuonBarrelParams> newObject(const std::string& objectKey, const L1TMuonBarrelParamsO2ORcd& record) override ;

    L1TMuonBarrelParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonBarrelParamsOnlineProd(void){}
};

L1TMuonBarrelParamsOnlineProd::L1TMuonBarrelParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonBarrelParamsO2ORcd,L1TMuonBarrelParams>(iConfig) {}

boost::shared_ptr<L1TMuonBarrelParams> L1TMuonBarrelParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonBarrelParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TMuonBarrelParamsRcd& baseRcd = record.template getRecord< L1TMuonBarrelParamsRcd >() ;
    edm::ESHandle< L1TMuonBarrelParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if (objectKey.empty()) {
        edm::LogInfo( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Key is empty, returning empty L1TMuonBarrelParams";
        return boost::shared_ptr< L1TMuonBarrelParams > ( new L1TMuonBarrelParams( *(baseSettings.product()) ) );
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );


    std::string stage2Schema = "CMS_TRG_L1_CONF" ;
    edm::LogInfo( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Producing L1TMuonBarrelParams with TSC key =" << tscKey << " and RS key = " << rsKey ;

        // first, find keys for the algo and RS tables

        // ALGO and HW
        std::vector< std::string > queryStrings ;
        queryStrings.push_back( "ALGO" ) ;
        queryStrings.push_back( "HW"   ) ;

        std::string algo_key, hw_key;

        // select ALGO,HW from CMS_TRG_L1_CONF.BMTF_KEYS where ID = tscKey ;
        l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_KEYS",
                                     "BMTF_KEYS.ID",
                                     m_omdsReader.singleAttribute(tscKey)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O" ) << "Cannot get BMTF_KEYS.{ALGO,HW}" ;
            return boost::shared_ptr< L1TMuonBarrelParams > ( new L1TMuonBarrelParams( *(baseSettings.product()) ) );
        }

        if( !queryResult.fillVariable( "ALGO", algo_key) ) algo_key = "";
        if( !queryResult.fillVariable( "HW",   hw_key  ) ) hw_key   = "";


        // RS
        queryStrings.clear() ;
        queryStrings.push_back( "MP7"    ) ;
        queryStrings.push_back( "DAQTTC" ) ;

        std::string rs_mp7_key, rs_amc13_key;

        // select RS from CMS_TRG_L1_CONF.BMTF_RS_KEYS where ID = rsKey ;
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_RS_KEYS",
                                     "BMTF_RS_KEYS.ID",
                                     m_omdsReader.singleAttribute(rsKey)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O" ) << "Cannot get BMTF_RS_KEYS.{MP7,DAQTTC}" ;
            return boost::shared_ptr< L1TMuonBarrelParams > ( new L1TMuonBarrelParams( *(baseSettings.product()) ) );
        }

        if( !queryResult.fillVariable( "MP7",    rs_mp7_key  ) ) rs_mp7_key   = "";
        if( !queryResult.fillVariable( "DAQTTC", rs_amc13_key) ) rs_amc13_key = "";


        // At this point we have four keys: one ALGO key, one HW key, and two RS keys; now query the payloads for these keys
        // Now querry the actual payloads
        enum {kALGO=0, kRS, kHW, NUM_TYPES};
        std::map<std::string,std::string> payloads[NUM_TYPES];  // associates key -> XML payload for a given type of payloads
        std::string xmlPayload;

        queryStrings.clear();
        queryStrings.push_back( "CONF" );

        // query ALGO configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_ALGO",
                                     "BMTF_ALGO.ID",
                                     m_omdsReader.singleAttribute(algo_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Cannot get BMTF_ALGO.CONF for ID="<<algo_key;
            return boost::shared_ptr< L1TMuonBarrelParams >( new L1TMuonBarrelParams( *(baseSettings.product()) ) ) ;
        }

        if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
        // remember ALGO configuration
        payloads[kALGO][algo_key] = xmlPayload;

        // query HW configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_HW",
                                     "BMTF_HW.ID",
                                     m_omdsReader.singleAttribute(hw_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Cannot get BMTF_HW.CONF for ID="<<hw_key;
            return boost::shared_ptr< L1TMuonBarrelParams >( new L1TMuonBarrelParams( *(baseSettings.product()) ) ) ;
        }

        if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
        // remember HW configuration
        payloads[kHW][hw_key] = xmlPayload;

        // query MP7 RS configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_RS",
                                     "BMTF_RS.ID",
                                     m_omdsReader.singleAttribute(rs_mp7_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Cannot get BMTF_RS.CONF for ID="<<rs_mp7_key;
            return boost::shared_ptr< L1TMuonBarrelParams >( new L1TMuonBarrelParams( *(baseSettings.product()) ) ) ;
        }

        if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
        // remember MP7 RS configuration
        payloads[kRS][rs_mp7_key] = xmlPayload;

        // query AMC13 RS configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "BMTF_RS",
                                     "BMTF_RS.ID",
                                     m_omdsReader.singleAttribute(rs_amc13_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonBarrelParamsOnlineProd" ) << "Cannot get BMTF_RS.CONF for ID="<<rs_amc13_key;
            return boost::shared_ptr< L1TMuonBarrelParams >( new L1TMuonBarrelParams( *(baseSettings.product()) ) ) ;
        }

        if( !queryResult.fillVariable( "CONF", xmlPayload ) ) xmlPayload = "";
        // remember AMC13 RS configuration
        payloads[kRS][rs_amc13_key] = xmlPayload;

// for debugging dump the configs to local files
for(auto &conf : payloads[kHW]){ 
    std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
    output<<conf.second;
    output.close();
}
for(auto &conf : payloads[kALGO]){ 
    std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
    output<<conf.second;
    output.close();
}
for(auto &conf : payloads[kRS]){ 
    std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
    output<<conf.second;
    output.close();
}

        // finally, push all payloads to the XML parser and construct the TrigSystem objects with each of those
        l1t::XmlConfigReader xmlRdr;
        l1t::TrigSystem parsedXMLs;
//        parsedXMLs.addProcRole("processors", "procMP7");
        // HW settings should always go first
        for(auto &conf : payloads[ kHW ]){
            xmlRdr.readDOMFromString( conf.second );
            xmlRdr.readRootElement  ( parsedXMLs  );
        }
        // now let's parse ALGO and then RS settings 
        for(auto &conf : payloads[ kALGO ]){
            xmlRdr.readDOMFromString( conf.second );
            xmlRdr.readRootElement  ( parsedXMLs  );
        }
        for(auto &conf : payloads[ kRS ]){
            xmlRdr.readDOMFromString( conf.second );
            xmlRdr.readRootElement  ( parsedXMLs  );
        }
        parsedXMLs.setConfigured();

        // for debugging also dump the configs to local files
        for(size_t type=0; type<NUM_TYPES; type++)
            for(auto &conf : payloads[ type ]){
                std::ofstream output(std::string("/tmp/").append(conf.first.substr(0,conf.first.find("/"))).append(".xml"));
                output<<conf.second;
                output.close();
            }

        L1TMuonBarrelParamsHelper m_params_helper(*(baseSettings.product()) );
        m_params_helper.configFromDB(parsedXMLs);
        boost::shared_ptr< L1TMuonBarrelParams > retval( new L1TMuonBarrelParams(m_params_helper) ) ;
 
        return retval;

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonBarrelParamsOnlineProd);
