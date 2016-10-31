#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

class L1TMuonOverlapParamsOnlineProxy : public edm::ESProducer {
private:
public:
    boost::shared_ptr<L1TMuonOverlapParams> produce(const L1TMuonOverlapParamsO2ORcd& record);

    L1TMuonOverlapParamsOnlineProxy(const edm::ParameterSet&);
    ~L1TMuonOverlapParamsOnlineProxy(void){}
};

L1TMuonOverlapParamsOnlineProxy::L1TMuonOverlapParamsOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
    setWhatProduced(this);
}

boost::shared_ptr<L1TMuonOverlapParams> L1TMuonOverlapParamsOnlineProxy::produce(const L1TMuonOverlapParamsO2ORcd& record) {

    const L1TMuonOverlapParamsRcd& baseRcd = record.template getRecord< L1TMuonOverlapParamsRcd >() ;
    edm::ESHandle< L1TMuonOverlapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;

    return boost::shared_ptr< L1TMuonOverlapParams > ( new L1TMuonOverlapParams( *(baseSettings.product()) ) );

}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsOnlineProxy);
