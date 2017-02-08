#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
#include <map>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsO2ORcd.h"
#include "L1Trigger/L1TMuonEndCap/interface/EndCapParamsHelper.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCommon/interface/TrigSystem.h"

class L1TMuonEndcapParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams> {
private:
public:
    virtual std::shared_ptr<L1TMuonEndCapParams> newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) override ;

    L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndcapParamsOnlineProd(void){}
};

L1TMuonEndcapParamsOnlineProd::L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams>(iConfig) {}

std::shared_ptr<L1TMuonEndCapParams> L1TMuonEndcapParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TMuonEndcapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndcapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndCapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if (objectKey.empty()) {
        edm::LogInfo( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Key is empty, returning empty L1TMuonEndcapParams";
        throw std::runtime_error("Empty objectKey");
//        return std::shared_ptr< L1TMuonEndCapParams >( new L1TMuonEndCapParams( *(baseSettings.product()) ) );
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );


    std::string stage2Schema = "CMS_TRG_L1_CONF" ;
    edm::LogInfo( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Producing L1TMuonEndcapParams with TSC key =" << tscKey << " and RS key = " << rsKey ;

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
                                     "EMTF_KEYS",
                                     "EMTF_KEYS.ID",
                                     m_omdsReader.singleAttribute(tscKey)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O" ) << "Cannot get EMTF_KEYS.{ALGO,HW}" ;
            throw std::runtime_error("Broken key");
            //return std::shared_ptr< L1TMuonEndCapParams >( new L1TMuonEndCapParams( *(baseSettings.product()) ) );
        }

        if( !queryResult.fillVariable( "ALGO", algo_key) ) algo_key = "";
        if( !queryResult.fillVariable( "HW",   hw_key  ) ) hw_key   = "";

    // all EMTF ALGO keys before "EMTF_ALGO_BASE/v6" are broken and will crash the code below, let's promote them to v6
    size_t pos = algo_key.find("EMTF_ALGO_BASE/v");
    if( pos != std::string::npos ){
        int v = atoi(algo_key.c_str()+16);
        if( v>0 && v<6 ){
            algo_key = "EMTF_ALGO_BASE/v6";
            edm::LogError( "L1-O2O" ) << "Inconsistent old ALGO key -> changing to " << algo_key ;
        }
    }
    // HW has to be consistent with the ALGO: promote everything to EMTF_HW/v8
    pos = hw_key.find("EMTF_HW/v");
    if( pos != std::string::npos ){
        int v = atoi(hw_key.c_str()+9);
        if( v>0 && v<8 ){
            hw_key = "EMTF_HW/v8";
            edm::LogError( "L1-O2O" ) << "Inconsistent old HW key -> changing to " << hw_key ;
        }
    }

        queryStrings.clear();
        queryStrings.push_back( "CONF" );

        std::string xmlHWpayload;

        // query HW configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "EMTF_HW",
                                     "EMTF_HW.ID",
                                     m_omdsReader.singleAttribute(hw_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Cannot get EMTF_HW.CONF for ID="<<hw_key;
            throw std::runtime_error("Broken key");
            //return std::shared_ptr< L1TMuonEndCapParams >( new L1TMuonEndCapParams( *(baseSettings.product()) ) );
        }

        if( !queryResult.fillVariable( "CONF", xmlHWpayload ) ) xmlHWpayload = "";

        // query ALGO configuration
        queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "EMTF_ALGO",
                                     "EMTF_ALGO.ID",
                                     m_omdsReader.singleAttribute(algo_key)
                                   ) ;

        if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
            edm::LogError( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Cannot get EMTF_ALGO.CONF for ID="<<algo_key;
            throw std::runtime_error("Broken key");
            //return std::shared_ptr< L1TMuonEndCapParams >( new L1TMuonEndCapParams( *(baseSettings.product()) ) );
        }

        std::string xmlALGOpayload;

        if( !queryResult.fillVariable( "CONF", xmlALGOpayload ) ) xmlALGOpayload = "";


        l1t::XmlConfigReader xmlRdr;
        l1t::TrigSystem trgSys;

        xmlRdr.readDOMFromString( xmlHWpayload );
        xmlRdr.readRootElement  ( trgSys       );

        xmlRdr.readDOMFromString( xmlALGOpayload );
        xmlRdr.readRootElement  ( trgSys         );

        trgSys.setConfigured();


        std::map<std::string, l1t::Setting> conf = trgSys.getSettings("EMTF-1"); // any processor

        std::string core_fwv = conf["core_firmware_version"].getValueAsStr();
        tm brokenTime;
        strptime(core_fwv.c_str(), "%Y-%m-%d %T", &brokenTime);
        time_t sinceEpoch = timegm(&brokenTime);

        l1t::EndCapParamsHelper data( new L1TMuonEndCapParams() );

        data.SetFirmwareVersion( sinceEpoch );
        data.SetPtAssignVersion( conf["pt_lut_version"].getValue<unsigned int>() );

        std::shared_ptr< L1TMuonEndCapParams > retval( data.getWriteInstance() );

    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapParamsOnlineProd);
