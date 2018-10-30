#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TMuonBarrelObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    void fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ) override ;

    L1TMuonBarrelObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TMuonBarrelObjectKeysOnlineProd(void) override{}
};

L1TMuonBarrelObjectKeysOnlineProd::L1TMuonBarrelObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TMuonBarrelObjectKeysOnlineProd::fillObjectKeys( L1TriggerKeyExt* pL1TriggerKey ){

    std::string BMTFKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kBMTF ) ;

    std::string stage2Schema = "CMS_TRG_L1_CONF" ;

    // simply assign the top level key to the record
    pL1TriggerKey->add( "L1TMuonBarrelParamsO2ORcd",
                        "L1TMuonBarrelParams",
                        BMTFKey) ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonBarrelObjectKeysOnlineProd);
