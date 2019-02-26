#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonEndCapObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:
    bool transactionSafe;
public:
    void fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ) override ;

    L1TMuonEndCapObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonEndCapObjectKeysOnlineProd(void) override{}
};

L1TMuonEndCapObjectKeysOnlineProd::L1TMuonEndCapObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig )
{
    transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}


void L1TMuonEndCapObjectKeysOnlineProd::fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ){

    std::string EMTFKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kEMTF ) ;

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndCapParamsO2ORcd",
                        "L1TMuonEndCapParams",
                        EMTFKey) ;

    std::string tscKey = EMTFKey.substr(0, EMTFKey.find(":") );

////////////////////////
// the block below reproduces L1TMuonEndCapParamsOnlineProd identically

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
        edm::LogError( "L1-O2O: L1TMuonEndCapObjectKeysOnlineProd" ) << e.what();
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: EMTF  | Faulty  | Broken key");
        else {
            edm::LogError( "L1-O2O: L1TMuonEndCapObjectKeysOnlineProd" ) << "forcing L1TMuonEndCapForest key to be = '7' (known to exist)";
            pL1TriggerKey->add( "L1TMuonEndCapForestO2ORcd",
                                "L1TMuonEndCapForest",
                                "7") ;
            return;
        }
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
        edm::LogError( "L1-O2O: L1TMuonEndCapObjectKeysOnlineProd" ) << e.what();
        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: EMTF  | Faulty  | Cannot parse XMLs");
        else {
            edm::LogError( "L1-O2O: L1TMuonEndCapObjectKeysOnlineProd" ) << "forcing L1TMuonEndCapForest key to be = '7' (known to exist)";
            pL1TriggerKey->add( "L1TMuonEndCapForestO2ORcd",
                                "L1TMuonEndCapForest",
                                "7") ;
            return;
        }
    }

    // Changed from "EMTF-1" to "EMTFp1", but is this backwards compatible? Does it need to be? - AWB 10.04.2018
    std::map<std::string, l1t::Parameter> conf = trgSys.getParameters("EMTFp1"); // any processor will do
    // if (conf still empty) conf = trgSys.getParameters("EMTF+1"); // Should add some conditional for backwards-compatibility

////////////////////////

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndCapForestO2ORcd",
                        "L1TMuonEndCapForest",
                        conf["pt_lut_version"].getValueAsStr()) ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapObjectKeysOnlineProd);
