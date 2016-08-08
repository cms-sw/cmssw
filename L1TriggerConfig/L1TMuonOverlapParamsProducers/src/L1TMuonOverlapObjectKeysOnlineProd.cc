#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TMuonOverlapObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    virtual void fillObjectKeys( ReturnType pL1TriggerKey ) override ;

    L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonOverlapObjectKeysOnlineProd(void){}
};

L1TMuonOverlapObjectKeysOnlineProd::L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TMuonOverlapObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey ){

    std::string OMTFKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kOMTF ) ;

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    if( OMTFKey.empty() ){
        edm::LogError( "L1-O2O: L1TMuonOverlapObjectKeysOnlineProd" ) << "Key is empty ... do nothing, but that'll probably crash things later on";
        return;
    }

    std::string tscKey = OMTFKey.substr(0, OMTFKey.find(":") );
    std::string  rsKey = OMTFKey.substr(   OMTFKey.find(":")+1, std::string::npos );

    std::vector< std::string > queryStrings ;
    queryStrings.push_back( "ALGO" ) ;

    std::string algo_key;

    // select ALGO from CMS_TRG_L1_CONF.OMTF_KEYS where ID = tscKey ;
    l1t::OMDSReader::QueryResults queryResult =
            m_omdsReader.basicQuery( queryStrings,
                                     stage2Schema,
                                     "OMTF_KEYS",
                                     "OMTF_KEYS.ID",
                                     m_omdsReader.singleAttribute(tscKey)
                                   ) ;

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 ){
        edm::LogError( "L1-O2O" ) << "Cannot get OMTF_KEYS.ALGO "<<" do nothing, but that'll probably crash things later on";
        return; 
    }

    if( !queryResult.fillVariable( "ALGO", algo_key) ) algo_key = "";

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonOverlapParamsO2ORcd",
                        "L1TMuonOverlapParams",
			algo_key) ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapObjectKeysOnlineProd);
