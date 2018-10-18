#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestO2ORcd.h"

class L1TMuonEndCapForestOnlineProxy : public edm::ESProducer {
public:
    std::unique_ptr<L1TMuonEndCapForest> produce(const L1TMuonEndCapForestO2ORcd& record);

    L1TMuonEndCapForestOnlineProxy(const edm::ParameterSet&);
    ~L1TMuonEndCapForestOnlineProxy(void) override{}
};

L1TMuonEndCapForestOnlineProxy::L1TMuonEndCapForestOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
    setWhatProduced(this);
}

std::unique_ptr<L1TMuonEndCapForest> L1TMuonEndCapForestOnlineProxy::produce(const L1TMuonEndCapForestO2ORcd& record) {

    const L1TMuonEndCapForestRcd& baseRcd = record.template getRecord< L1TMuonEndCapForestRcd >() ;
    edm::ESHandle< L1TMuonEndCapForest > baseSettings ;
    baseRcd.get( baseSettings ) ;

    return std::make_unique< L1TMuonEndCapForest >( *(baseSettings.product()) );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapForestOnlineProxy);
