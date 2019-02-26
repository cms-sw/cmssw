#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TCaloParamsObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    void fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ) override ;

    L1TCaloParamsObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TCaloParamsObjectKeysOnlineProd(void) override{}
};

L1TCaloParamsObjectKeysOnlineProd::L1TCaloParamsObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TCaloParamsObjectKeysOnlineProd::fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ){

    std::string CALOKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kCALO ) ;

    // simply assign the top level key to the record
    pL1TriggerKey->add( "L1TCaloParamsO2ORcd",
                        "CaloParams",
                        CALOKey ) ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TCaloParamsObjectKeysOnlineProd);
