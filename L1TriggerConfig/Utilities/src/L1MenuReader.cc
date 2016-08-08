#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1MenuReader : public edm::EDAnalyzer {
public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1MenuReader(const edm::ParameterSet&) : edm::EDAnalyzer(){}
    virtual ~L1MenuReader(void){}
};

void L1MenuReader::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    edm::ESHandle<L1GtTriggerMenu> handle1;
    evSetup.get<L1GtTriggerMenuRcd>().get( handle1 ) ;
    boost::shared_ptr<L1GtTriggerMenu> ptr1(new L1GtTriggerMenu(*(handle1.product ())));

    cout<<"L1GtTriggerMenu: "<<endl;
    cout<<" name: "<<ptr1->gtTriggerMenuName()<<endl;
    cout<<" iface: "<<ptr1->gtTriggerMenuInterface()<<endl;
    cout<<" implem: "<<ptr1->gtTriggerMenuImplementation()<<endl;
    cout<<" db_key: "<<ptr1->gtScaleDbKey()<<endl;

    cout<<" L1GtTriggerMenu: "<<endl;
    const std::vector<std::vector<L1GtMuonTemplate> >& muons = ptr1->vecMuonTemplate();
    int i=0, j=0;
    for(auto vec : muons){
        cout<<"  ["<<i<<"]"<<endl;
        i++;
        for(auto temp : vec){
            cout<<"   ["<<j<<"]";
            temp.print(cout);
            j++;
        }
    }

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1MenuReader);
