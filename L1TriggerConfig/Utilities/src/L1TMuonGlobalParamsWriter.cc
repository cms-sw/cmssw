#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonGlobalParamsWriter : public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TMuonGlobalParamsWriter(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1TMuonGlobalParamsWriter(void){}
};

void L1TMuonGlobalParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){
    edm::ESHandle<L1TMuonGlobalParams> handle1;
    evSetup.get<L1TMuonGlobalParamsO2ORcd>().get( handle1 ) ;
    boost::shared_ptr<L1TMuonGlobalParams> ptr1(new L1TMuonGlobalParams(*(handle1.product ())));

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if( poolDb.isAvailable() ){
        cond::Time_t firstSinceTime = poolDb->beginOfTime();
        poolDb->writeOne(ptr1.get(),firstSinceTime,"L1TMuonGlobalParamsO2ORcd");
    }

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonGlobalParamsWriter);

