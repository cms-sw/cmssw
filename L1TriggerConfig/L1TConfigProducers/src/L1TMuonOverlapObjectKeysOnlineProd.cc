#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TMuonOverlapObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:
    bool transactionSafe;
public:
    void fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ) override ;

    L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonOverlapObjectKeysOnlineProd(void) override{}
};

L1TMuonOverlapObjectKeysOnlineProd::L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
    transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}


void L1TMuonOverlapObjectKeysOnlineProd::fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ){

    std::string OMTFKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kOMTF ) ;

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    std::string tscKey = OMTFKey.substr(0, OMTFKey.find(":") );

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

    if( queryResult.queryFailed() || queryResult.numberRows() != 1 || !queryResult.fillVariable( "ALGO", algo_key) ){
        edm::LogError( "L1-O2O L1TMuonOverlapObjectKeysOnlineProd" ) << "Cannot get OMTF_KEYS.ALGO ";

        if( transactionSafe )
            throw std::runtime_error("SummaryForFunctionManager: OMTF  | Faulty  | Broken key");
        else {
            edm::LogError( "L1-O2O: L1TMuonOverlapObjectKeysOnlineProd" ) << "forcing L1TMuonOverlapParams key to be = 'OMTF_ALGO_EMPTY' (known to exist)";
            pL1TriggerKey->add( "L1TMuonOverlapParamsO2ORcd",
                                "L1TMuonOverlapParams",
                                "OMTF_ALGO_EMPTY") ;
            return;
        }
    }

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonOverlapParamsO2ORcd",
                        "L1TMuonOverlapParams",
			algo_key) ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapObjectKeysOnlineProd);
