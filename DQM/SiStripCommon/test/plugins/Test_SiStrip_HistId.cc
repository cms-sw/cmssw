
//
// Original Author:  Dorian Kcira
//         Created:  Thu Feb 23 18:50:29 CET 2006
//
//
// test HistId classes of SiStrip


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"



#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

class Test_SiStrip_HistId : public edm::EDAnalyzer {
   public:
      explicit Test_SiStrip_HistId(const edm::ParameterSet&);
      ~Test_SiStrip_HistId();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  EDGetTokenT<ExampleData> example_;
#endif

      // ----------member data ---------------------------
};

Test_SiStrip_HistId::Test_SiStrip_HistId(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  example_ = consumes<ExampleData>(edm::InputTag("data"));
#endif
}


Test_SiStrip_HistId::~Test_SiStrip_HistId()
{
}


void Test_SiStrip_HistId::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  edm::Handle<ExampleData> pIn;
  iEvent.getByToken(example_,pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   edm::ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

// use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  std::cout<<"------------------------------------"<<std::endl;
  std::string hid1 = hidmanager.createHistoId("Cluster Distribution","det",2345698);
  std::cout<<"created hid1: >>"<<hid1<<"<<"<<std::endl;
  std::string hid2 = hidmanager.createHistoId("Cluster Distribution","fec",1234);
  std::cout<<"created hid2: >>"<<hid2<<"<<"<<std::endl;
  std::string hid3 = hidmanager.createHistoId("Cluster Distribution","fed",5678);
  std::cout<<"created hid3: >>"<<hid3<<"<<"<<std::endl;
  std::cout<<"------------------------------------"<<std::endl;
  std::cout<<"hid1 component id / component type: "<<hidmanager.getComponentId(hid1)<<" / "<<hidmanager.getComponentType(hid1)<<std::endl;
  std::cout<<"hid2 component id / component type: "<<hidmanager.getComponentId(hid2)<<" / "<<hidmanager.getComponentType(hid2)<<std::endl;
  std::cout<<"hid3 component id / component type: "<<hidmanager.getComponentId(hid3)<<" / "<<hidmanager.getComponentType(hid3)<<std::endl;
  std::cout<<"------------------------------------"<<std::endl;
  std::string hid4="just for_testing% _#31";
  std::cout<<"hid4="<<hid4<<std::endl;
  std::cout<<"hid4 component id / component type: "<<hidmanager.getComponentId(hid4)<<" / "<<hidmanager.getComponentType(hid4)<<std::endl;
  std::cout<<"------------------------------------"<<std::endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Test_SiStrip_HistId);
