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

    // simply assign the algo key to the record
    pL1TriggerKey->add( "L1TMuonEndcapParamsO2ORcd",
                        "L1TMuonEndCapParams",
                        EMTFKey) ;

}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapObjectKeysOnlineProd);
