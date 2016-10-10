#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"
//#include "CondFormats/L1TObjects/interface/L1TMuonEndcapParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TEndcapReader: public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TEndcapReader(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1TEndcapReader(void){}
};

void L1TEndcapReader::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){
/*
    edm::ESHandle<L1TMuonEndcapParams> handle1;
    evSetup.get<L1TMuonEndcapParamsRcd>().get( handle1 ) ;
    boost::shared_ptr<L1TMuonEndcapParams> ptr1(new L1TMuonEndcapParams(*(handle1.product ())));

    cout<<"L1TMuonEndcapParams: "<<endl;
    ptr1->print(cout);

    // a more comprehensive output:
    std::vector <std::pair<int,EndCapForest*> > forests = ptr1->getPtForests();
    for(auto & forest : forests){
        cout<<" i="<<forest.first;
        for(unsigned int tr=0; tr<forest.second->size(); tr++){
            cout<<"  tree ptr="<<hex<<forest.second->getTree(tr)<<dec<<endl;
        }
    }
*/
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TEndcapReader);

