#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TMuonGlobalObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    void fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ) override ;

    L1TMuonGlobalObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonGlobalObjectKeysOnlineProd(void) override{}
};

L1TMuonGlobalObjectKeysOnlineProd::L1TMuonGlobalObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TMuonGlobalObjectKeysOnlineProd::fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ){

    std::string uGMTKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kuGMT ) ;

    // simply assign the top level key to the record
    pL1TriggerKey->add( "L1TMuonGlobalParamsO2ORcd",
                        "L1TMuonGlobalParams",
                        uGMTKey) ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonGlobalObjectKeysOnlineProd);
