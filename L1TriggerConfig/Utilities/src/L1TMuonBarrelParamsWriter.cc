#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonBarrelParamsWriter : public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TMuonBarrelParamsWriter(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1TMuonBarrelParamsWriter(void){}
};

void L1TMuonBarrelParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){
    edm::ESHandle<L1TMuonBarrelParams> handle1;
    evSetup.get<L1TMuonBarrelParamsO2ORcd>().get( handle1 ) ;
    boost::shared_ptr<L1TMuonBarrelParams> ptr1(new L1TMuonBarrelParams(*(handle1.product ())));

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if( poolDb.isAvailable() ){
        cond::Time_t firstSinceTime = poolDb->beginOfTime();
        poolDb->writeOne(ptr1.get(),firstSinceTime,"L1TMuonBarrelParamsO2ORcd");
    }

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonBarrelParamsWriter);

