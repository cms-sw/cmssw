#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsO2ORcd.h"

class L1TMuonEndcapParamsOnlineProxy : public edm::ESProducer {
private:
public:
    boost::shared_ptr<L1TMuonEndCapParams> produce(const L1TMuonEndcapParamsO2ORcd& record);

    L1TMuonEndcapParamsOnlineProxy(const edm::ParameterSet&);
    ~L1TMuonEndcapParamsOnlineProxy(void){}
};

L1TMuonEndcapParamsOnlineProxy::L1TMuonEndcapParamsOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
    setWhatProduced(this);
}

boost::shared_ptr<L1TMuonEndCapParams> L1TMuonEndcapParamsOnlineProxy::produce(const L1TMuonEndcapParamsO2ORcd& record) {
/*
    const L1TMuonEndcapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndcapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndcapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;

    return boost::shared_ptr< L1TMuonEndcapParams > ( new L1TMuonEndcapParams( *(baseSettings.product()) ) );
*/
    return boost::shared_ptr< L1TMuonEndCapParams > ( new L1TMuonEndCapParams() );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapParamsOnlineProxy);
