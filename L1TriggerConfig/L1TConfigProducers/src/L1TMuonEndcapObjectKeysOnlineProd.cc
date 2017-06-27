#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonEndcapObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    virtual void fillObjectKeys( ReturnType pL1TriggerKey ) override ;

    L1TMuonEndcapObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndcapObjectKeysOnlineProd(void){}
};

L1TMuonEndcapObjectKeysOnlineProd::L1TMuonEndcapObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TMuonEndcapObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey ){

    std::string EMTFKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kEMTF ) ;

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndcapParamsO2ORcd",
                        "L1TMuonEndCapParams",
                        EMTFKey) ;

    std::string tscKey = EMTFKey.substr(0, EMTFKey.find(":") );
    std::string  rsKey = EMTFKey.substr(   EMTFKey.find(":")+1, std::string::npos );

////////////////////////
// the block below reproduces L1TMuonEndcapParamsOnlineProd identically

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

    l1t::XmlConfigParser xmlRdr;
    l1t::TriggerSystem trgSys;

    xmlRdr.readDOMFromString( hw_payload );
    xmlRdr.readRootElement  ( trgSys     );

    xmlRdr.readDOMFromString( algo_payload );
    xmlRdr.readRootElement  ( trgSys       );

    trgSys.setConfigured();

    std::map<std::string, l1t::Parameter> conf = trgSys.getParameters("EMTF-1"); // any processor will do

////////////////////////

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndcapForestO2ORcd",
                        "L1TMuonEndCapForest",
                        conf["pt_lut_version"].getValueAsStr()) ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapObjectKeysOnlineProd);
