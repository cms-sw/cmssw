#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <map>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonEndCapParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonEndCapParamsO2ORcd,L1TMuonEndCapParams> {
private:
    bool transactionSafe;
public:
    std::shared_ptr<L1TMuonEndCapParams> newObject(const std::string& objectKey, const L1TMuonEndCapParamsO2ORcd& record) override ;

    L1TMuonEndCapParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndCapParamsOnlineProd(void) override{}
};

L1TMuonEndCapParamsOnlineProd::L1TMuonEndCapParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonEndCapParamsO2ORcd,L1TMuonEndCapParams>(iConfig){
    transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

std::shared_ptr<L1TMuonEndCapParams> L1TMuonEndCapParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonEndCapParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TMuonEndCapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndCapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndCapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if (objectKey.empty()) {
        edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "Key is empty";
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: BMTF  | Faulty  | Empty objectKey");
        else {
            edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "returning unmodified prototype of L1TMuonEndCapParams";
            return std::make_shared< L1TMuonEndCapParams >( *(baseSettings.product()) ) ;
        }
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );

    edm::LogInfo( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "Producing L1TMuonEndCapParams with TSC key = " << tscKey << " and RS key = " << rsKey ;

    std::string algo_key, hw_key;
    std::string algo_payload, hw_payload;
    try {
        std::map<std::string,std::string> keys =
            l1t::OnlineDBqueryHelper::fetch( {"HW","ALGO"},
                                             "EMTF_KEYS",
                                             tscKey,
                                             m_omdsReader
                                           );

        hw_key   = keys["HW"];
        algo_key = keys["ALGO"];

        hw_payload = l1t::OnlineDBqueryHelper::fetch( {"CONF"},
                                                      "EMTF_CLOBS",
                                                       hw_key,
                                                       m_omdsReader
                                                    ) ["CONF"];

        algo_payload = l1t::OnlineDBqueryHelper::fetch( {"CONF"},
                                                        "EMTF_CLOBS",
                                                         algo_key,
                                                         m_omdsReader
                                                      ) ["CONF"];

    } catch ( std::runtime_error &e ) {
        edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << e.what();
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: EMTF  | Faulty  | Broken key");
        else {
            edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "returning unmodified prototype of L1TMuonEndCapParams";
            return std::make_shared< L1TMuonEndCapParams >( *(baseSettings.product()) ) ;
        }
    }

    // for debugging purposes dump the configs to local files
    {
        std::ofstream output(std::string("/tmp/").append(hw_key.substr(0,hw_key.find("/"))).append(".xml"));
        output << hw_payload;
        output.close();
    }
    {
        std::ofstream output(std::string("/tmp/").append(algo_key.substr(0,algo_key.find("/"))).append(".xml"));
        output << algo_payload;
        output.close();
    }

    l1t::XmlConfigParser xmlRdr;
    l1t::TriggerSystem trgSys;

    try {
        xmlRdr.readDOMFromString( hw_payload );
        xmlRdr.readRootElement  ( trgSys     );

        xmlRdr.readDOMFromString( algo_payload );
        xmlRdr.readRootElement  ( trgSys       );

        trgSys.setConfigured();
    } catch ( std::runtime_error &e ) {
        edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << e.what();
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: EMTF  | Faulty  | Cannot parse XMLs");
        else {
            edm::LogError( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "returning unmodified prototype of L1TMuonEndCapParams";
            return std::make_shared< L1TMuonEndCapParams >( *(baseSettings.product()) ) ;
        }
    }

    // Changed from "EMTF-1" to "EMTFp1", but is this backwards compatible? Does it need to be? - AWB 10.04.2018
    std::map<std::string, l1t::Parameter> conf = trgSys.getParameters("EMTFp1"); // any processor will do
    // if (conf still empty) conf = trgSys.getParameters("EMTF+1"); // Should add some conditional for backwards-compatibility

    std::string core_fwv = conf["core_firmware_version"].getValueAsStr();
    tm brokenTime;
    strptime(core_fwv.c_str(), "%Y-%m-%d %T", &brokenTime);
    time_t fw_sinceEpoch = timegm(&brokenTime);

//    std::string pclut_v = conf["pc_lut_version"].getValueAsStr();
//    strptime(pclut_v.c_str(), "%Y-%m-%d", &brokenTime);
//    time_t pclut_sinceEpoch = timegm(&brokenTime);

    std::shared_ptr< L1TMuonEndCapParams > retval( new L1TMuonEndCapParams() ); 
    
    retval->firmwareVersion_ = fw_sinceEpoch;
    retval->PtAssignVersion_ = conf["pt_lut_version"].getValue<unsigned int>();
    retval->PhiMatchWindowSt1_ = 1; //pclut_sinceEpoch;

    edm::LogInfo( "L1-O2O: L1TMuonEndCapParamsOnlineProd" ) << "SummaryForFunctionManager: EMTF  | OK      | All looks good";
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapParamsOnlineProd);
