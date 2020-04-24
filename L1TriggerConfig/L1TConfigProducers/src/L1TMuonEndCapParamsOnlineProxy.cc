#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsO2ORcd.h"

class L1TMuonEndCapParamsOnlineProxy : public edm::ESProducer {
private:
    unsigned int PtAssignVersion, firmwareVersion, changeDate;
public:
    std::shared_ptr<L1TMuonEndCapParams> produce(const L1TMuonEndCapParamsO2ORcd& record);

    L1TMuonEndCapParamsOnlineProxy(const edm::ParameterSet&);
    ~L1TMuonEndCapParamsOnlineProxy(void) override{}
};

L1TMuonEndCapParamsOnlineProxy::L1TMuonEndCapParamsOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
    setWhatProduced(this);
    PtAssignVersion = iConfig.getUntrackedParameter<unsigned int>("PtAssignVersion", 1);
    firmwareVersion = iConfig.getUntrackedParameter<unsigned int>("firmwareVersion", 1);
    changeDate      = iConfig.getUntrackedParameter<unsigned int>("changeDate",      1);
}

std::shared_ptr<L1TMuonEndCapParams> L1TMuonEndCapParamsOnlineProxy::produce(const L1TMuonEndCapParamsO2ORcd& record) {
/*
    const L1TMuonEndCapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndCapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndCapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;

    return boost::shared_ptr< L1TMuonEndCapParams > ( new L1TMuonEndCapParams( *(baseSettings.product()) ) );
*/
    std::shared_ptr< L1TMuonEndCapParams > retval = std::make_shared< L1TMuonEndCapParams>();

    retval->PtAssignVersion_ = PtAssignVersion;
    retval->firmwareVersion_ = firmwareVersion; 
    retval->PhiMatchWindowSt1_ = changeDate; // This should be set to PrimConvVersion - AWB 13.06.17
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapParamsOnlineProxy);
