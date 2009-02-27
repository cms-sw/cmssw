#include <iostream>

#include "DQMOffline/EGamma/interface/PhotonDataCertification.h"

/**\class PhotonDataCertification 
*/
//
// Original Author:  Louis James Antonelli
//         Created:  Thu Jan 22 13:42:28CET 2009
// $Id: PhotonDataCertification.cc,v 1.2 2009/02/26 15:53:26 lantonel Exp $
//
//
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"





using namespace std;

PhotonDataCertification::PhotonDataCertification(const edm::ParameterSet& pset)

{

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  parameters_ = pset;


}


PhotonDataCertification::~PhotonDataCertification()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PhotonDataCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   

   //std::cout << "bla" << std::endl;


}


// ------------ method called once each job just before starting event loop  ------------
void 
PhotonDataCertification::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PhotonDataCertification::endJob() {

  //  MonitorElement * RefHist = new MonitorElement(*(dbe_->get("Egamma/PhotonAnalyzer/BackgroundPhotons/Et above 0 GeV/r9AllEcal")));
//   TH1F * RefHistTH1 = RefHist->getTH1F();
//   MonitorElement * TestHist = new MonitorElement(*(dbe_->get("Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/r9AllEcal")));
//   TH1F * TestHistTH1 = TestHist->getTH1F();
//   double KolTest;

//   KolTest = TestHistTH1->KolmogorovTest(RefHistTH1);

//   cout << KolTest << endl;


}
