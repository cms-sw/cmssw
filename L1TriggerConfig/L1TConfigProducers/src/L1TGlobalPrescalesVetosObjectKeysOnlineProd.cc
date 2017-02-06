#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class L1TGlobalPrescalesVetosObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:

public:
    virtual void fillObjectKeys( ReturnType pL1TriggerKey ) override ;

    L1TGlobalPrescalesVetosObjectKeysOnlineProd(const edm::ParameterSet&);
    ~L1TGlobalPrescalesVetosObjectKeysOnlineProd(void){}
};

L1TGlobalPrescalesVetosObjectKeysOnlineProd::L1TGlobalPrescalesVetosObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
  : L1ObjectKeysOnlineProdBaseExt( iConfig ){
}


void L1TGlobalPrescalesVetosObjectKeysOnlineProd::fillObjectKeys( ReturnType pL1TriggerKey ){

    std::string uGTKey = pL1TriggerKey->subsystemKey( L1TriggerKeyExt::kuGT ) ;

    std::string uGTrsKey = uGTKey.substr( uGTKey.find(":")+1 );

    pL1TriggerKey->add( "L1TGlobalPrescalesVetosO2ORcd",
                        "L1TGlobalPrescalesVetos",
                         uGTrsKey) ;
}


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosObjectKeysOnlineProd);
