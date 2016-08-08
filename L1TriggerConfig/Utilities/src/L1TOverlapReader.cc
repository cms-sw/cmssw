#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
///#include "CondFormats/DataRecord/interface/L1TMuonOverlapPatternParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TOverlapReader: public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TOverlapReader(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1TOverlapReader(void){}
};

void L1TOverlapReader::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    // Pull the config from the ES
    edm::ESHandle<L1TMuonOverlapParams> handle1;
    evSetup.get<L1TMuonOverlapParamsRcd>().get( handle1 ) ;
    boost::shared_ptr<L1TMuonOverlapParams> ptr1(new L1TMuonOverlapParams(*(handle1.product ())));

    // Separately pull patterns from the ES
///    edm::ESHandle<L1TMuonOverlapParams> handle2;
///    evSetup.get<L1TMuonOverlapPatternParamsRcd>().get( handle2 ) ; // note: another data record type here
///    boost::shared_ptr<L1TMuonOverlapParams> ptr2(new L1TMuonOverlapParams(*(handle2.product ())));

    cout<<"Some fields in L1TMuonOverlapParams: "<<endl;

///    cout<<"nGoldenPatterns() = "<<ptr2->nGoldenPatterns()<<endl;

    const std::vector<int>* gp = ptr1->generalParams();
    cout<<"number of general parameters: = "<<gp->size()<<endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TOverlapReader);

