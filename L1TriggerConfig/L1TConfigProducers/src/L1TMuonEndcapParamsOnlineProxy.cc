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
    unsigned int PtAssignVersion, firmwareVersion, changeDate;
public:
    std::shared_ptr<L1TMuonEndCapParams> produce(const L1TMuonEndcapParamsO2ORcd& record);

    L1TMuonEndcapParamsOnlineProxy(const edm::ParameterSet&);
    ~L1TMuonEndcapParamsOnlineProxy(void){}
};

L1TMuonEndcapParamsOnlineProxy::L1TMuonEndcapParamsOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
    setWhatProduced(this);
    PtAssignVersion = iConfig.getUntrackedParameter<unsigned int>("PtAssignVersion", 1);
    firmwareVersion = iConfig.getUntrackedParameter<unsigned int>("firmwareVersion", 1);
    changeDate      = iConfig.getUntrackedParameter<unsigned int>("changeDate",      1);
}

std::shared_ptr<L1TMuonEndCapParams> L1TMuonEndcapParamsOnlineProxy::produce(const L1TMuonEndcapParamsO2ORcd& record) {
/*
    const L1TMuonEndcapParamsRcd& baseRcd = record.template getRecord< L1TMuonEndcapParamsRcd >() ;
    edm::ESHandle< L1TMuonEndcapParams > baseSettings ;
    baseRcd.get( baseSettings ) ;

    return boost::shared_ptr< L1TMuonEndcapParams > ( new L1TMuonEndcapParams( *(baseSettings.product()) ) );
*/
    std::shared_ptr< L1TMuonEndCapParams > retval = std::make_shared< L1TMuonEndCapParams>();

    retval->PtAssignVersion_ = PtAssignVersion;
    retval->firmwareVersion_ = firmwareVersion; 
    // retval->PhiMatchWindowSt1_ = changeDate;  // This should be set to PrimConvVersion - AWB 13.06.17
    return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndcapParamsOnlineProxy);
