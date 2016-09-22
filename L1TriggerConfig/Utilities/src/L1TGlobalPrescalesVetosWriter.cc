#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TGlobalPrescalesVetosWriter : public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TGlobalPrescalesVetosWriter(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1TGlobalPrescalesVetosWriter(void){}
};

void L1TGlobalPrescalesVetosWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){
    edm::ESHandle<L1TGlobalPrescalesVetos> handle1;
    evSetup.get<L1TGlobalPrescalesVetosO2ORcd>().get( handle1 ) ;
    boost::shared_ptr<L1TGlobalPrescalesVetos> ptr1(new L1TGlobalPrescalesVetos(*(handle1.product ())));

    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if( poolDb.isAvailable() ){
        cond::Time_t firstSinceTime = poolDb->beginOfTime();
        poolDb->writeOne(ptr1.get(),firstSinceTime,"L1TGlobalPrescalesVetosO2ORcd");
    }

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TGlobalPrescalesVetosWriter);

