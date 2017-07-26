#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
#include <map>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsO2ORcd.h"
#include "L1Trigger/L1TMuonEndCap/interface/EndCapParamsHelper.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonEndcapParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams> {
private:
public:
    virtual std::shared_ptr<L1TMuonEndCapParams> newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) override ;

    L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndcapParamsOnlineProd(void){}
};

L1TMuonEndcapParamsOnlineProd::L1TMuonEndcapParamsOnlineProd(const edm::ParameterSet& iConfig) : L1ConfigOnlineProdBaseExt<L1TMuonEndcapParamsO2ORcd,L1TMuonEndCapParams>(iConfig){}

std::shared_ptr<L1TMuonEndCapParams> L1TMuonEndcapParamsOnlineProd::newObject(const std::string& objectKey, const L1TMuonEndcapParamsO2ORcd& record) {
    using namespace edm::es;

    const L1TMuonEndcapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndcapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndCapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;


    if (objectKey.empty()) {
        edm::LogError( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Key is empty, returning empty L1TMuonEndcapParams";
        throw std::runtime_error("Empty objectKey");
    }

    std::string tscKey = objectKey.substr(0, objectKey.find(":") );
    std::string  rsKey = objectKey.substr(   objectKey.find(":")+1, std::string::npos );

    edm::LogInfo( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << "Producing L1TMuonEndcapParams with TSC key = " << tscKey << " and RS key = " << rsKey ;

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
        edm::LogError( "L1-O2O: L1TMuonEndcapParamsOnlineProd" ) << e.what();
        throw std::runtime_error("Broken key");
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

    xmlRdr.readDOMFromString( hw_payload );
    xmlRdr.readRootElement  ( trgSys     );

    xmlRdr.readDOMFromString( algo_payload );
    xmlRdr.readRootElement  ( trgSys       );

    trgSys.setConfigured();

    std::map<std::string, l1t::Parameter> conf = trgSys.getParameters("EMTF-1"); // any processor will do

    std::string core_fwv = conf["core_firmware_version"].getValueAsStr();
    tm brokenTime;
    strptime(core_fwv.c_str(), "%Y-%m-%d %T", &brokenTime);
    time_t fw_sinceEpoch = timegm(&brokenTime);

    std::string pclut_v = conf["pc_lut_version"].getValueAsStr();
    strptime(pclut_v.c_str(), "%Y-%m-%d", &brokenTime);
    time_t pclut_sinceEpoch = timegm(&brokenTime);

    l1t::EndCapParamsHelper data( new L1TMuonEndCapParams() );

    data.SetFirmwareVersion( fw_sinceEpoch );
    data.SetPtAssignVersion( conf["pt_lut_version"].getValue<unsigned int>() );
    data.SetPrimConvVersion( pclut_sinceEpoch );

    std::shared_ptr< L1TMuonEndCapParams > retval( data.getWriteInstance() ); 

    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapParamsOnlineProd);
