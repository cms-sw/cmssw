#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

///#include "L1Trigger/L1TMuonEndCap/interface/ForestHelper.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
//#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1TMuonEndCapParamsViewer : public edm::EDAnalyzer {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonEndCapParamsViewer(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1TMuonEndCapParamsViewer(void) override {}
};

void L1TMuonEndCapParamsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonEndCapParams> handle1;
  evSetup.get<L1TMuonEndCapParamsRcd>().get(handle1);
  //    evSetup.get<L1TMuonEndCapParamsRcd>().get( handle1 ) ;
  std::shared_ptr<L1TMuonEndCapParams> ptr1(new L1TMuonEndCapParams(*(handle1.product())));

  cout << "L1TMuonEndCapParams: " << endl;
  cout << " PtAssignVersion_ = " << ptr1->PtAssignVersion_ << endl;
  cout << " firmwareVersion_ = " << ptr1->firmwareVersion_ << endl;
  cout << " PhiMatchWindowSt1_ = " << ptr1->PhiMatchWindowSt1_ << endl;
  cout << " PhiMatchWindowSt2_ = " << ptr1->PhiMatchWindowSt2_ << endl;
  cout << " PhiMatchWindowSt3_ = " << ptr1->PhiMatchWindowSt3_ << endl;
  cout << " PhiMatchWindowSt4_ = " << ptr1->PhiMatchWindowSt4_ << endl;

  ///    edm::ESHandle<L1TMuonEndCapForest> handle2;
  ///    evSetup.get<L1TMuonEndCapForestRcd>().get( handle2 ) ;
  ///    std::shared_ptr<L1TMuonEndCapForest> ptr2(new L1TMuonEndCapForest(*(handle2.product ())));
  ///
  ///    cout<<"L1TMuonEndCapForest: "<<endl;
  ///    l1t::ForestHelper fhelp( ptr2.get() );
  ///    fhelp.print( cout );
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonEndCapParamsViewer);
