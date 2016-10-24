#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    if( EMTFKey.empty() ){
        edm::LogError( "L1-O2O: L1TMuonEndcapObjectKeysOnlineProd" ) << "Key is empty ... do nothing, but that'll probably crash things later on";
        return;
    }

    std::string tscKey = EMTFKey.substr(0, EMTFKey.find(":") );
    std::string  rsKey = EMTFKey.substr(   EMTFKey.find(":")+1, std::string::npos );

    std::vector< std::string > queryStrings ;
    queryStrings.push_back( "ALGO" ) ;

    std::string algo_key;

    // select ALGO from CMS_TRG_L1_CONF.EMTF_KEYS where ID = tscKey ;
    l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "EMTF_KEYS",
                                     "EMTF_KEYS.ID",
                                     m_omdsReader.singleAttribute(tscKey)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O" ) << "Cannot get EMTF_KEYS.ALGO "<<" do nothing, but that'll probably crash things later on";
        return;
    }

    if( !queryResult.fillVariable( "ALGO", algo_key) ) algo_key = "";

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndcapParamsO2ORcd",
                        "L1TMuonEndCapParams",
                        algo_key) ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapObjectKeysOnlineProd);
