// -*- C++ -*-
//
// Package:    DQMOffline/JetMET
// Class:      DataCertificationJetMET
// 
/**\class DataCertificationJetMET DataCertificationJetMET.cc DQMOffline/JetMET/src/DataCertificationJetMET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Frank Chlebana"
//         Created:  Sun Oct  5 13:57:25 CDT 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

// #include "DQMOffline/JetMET/interface/DataCertificationJetMET.h"


//
// class decleration
//

class DataCertificationJetMET : public edm::EDAnalyzer {
   public:
      explicit DataCertificationJetMET(const edm::ParameterSet&);
      ~DataCertificationJetMET();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  edm::ParameterSet conf_;
  DQMStore * dbe;
  edm::Service<TFileService> fs_;


};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DataCertificationJetMET::DataCertificationJetMET(const edm::ParameterSet& iConfig):conf_(iConfig)

{
  // now do what ever initialization is needed

}


DataCertificationJetMET::~DataCertificationJetMET()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DataCertificationJetMET::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
#endif


}


// ------------ method called once each job just before starting event loop  ------------
void 
DataCertificationJetMET::beginJob(const edm::EventSetup&)
{
  std::string filename = conf_.getUntrackedParameter<std::string>("fileName");
  std::cout << "FileName = " << filename << std::endl;

  //  DQMStore * dbe;

  dbe = edm::Service<DQMStore>().operator->();
  dbe->open(filename);

  //  dbe = edm::Service<DQMStore>().operator->();
  //  dbe->open("/uscms/home/chlebana/DQM_V0001_R000063463__BeamHalo__BeamCommissioning08-PromptReco-v1__RECO.root");


//   // print histograms:
//   //  dbe->showDirStructure();
//   std::vector<MonitorElement*> mes = dbe->getAllContents("");
//   std::cout << "found " << mes.size() << " monitoring elements!" << std::endl;
//   TH1F *bla = fs_->make<TH1F>("bla","bla",256,0,256);
//   int totF;
//   for(std::vector<MonitorElement*>::const_iterator ime = mes.begin(); ime!=mes.end(); ++ime){
//     std::string name = (*ime)->getName();
//     if(name.find(tagname)>=name.size())
//       continue;
//     totF++;
    //std::cout << "hm found " << name << std::endl;
//     float filled=0;
//     float tot=0;
//     for(int ix=0; ix<=(*ime)->getNbinsX(); ++ix){
//       for(int iy=0; iy<=(*ime)->getNbinsY(); ++iy){
//      tot++;
//      if((*ime)->getBinContent(ix,iy)>0){
//        filled++;
//        std::cout << " " << (*ime)->getBinContent(ix,iy);
//        bla->Fill((*ime)->getBinContent(ix,iy));
//      }
//       }
//       std::cout << std::endl;
//     }
//     std::cout << name  << " " << filled/tot << std::endl;
//   }
//   std::cout << "tot " << totF << std::endl;


}

// ------------ method called once each job just after ending the event loop  ------------
void 
DataCertificationJetMET::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DataCertificationJetMET);
