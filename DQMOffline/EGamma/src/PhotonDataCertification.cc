#include <iostream>

#include "DQMOffline/EGamma/interface/PhotonDataCertification.h"

/**\class PhotonDataCertification 
*/
//
// Original Author:  Louis James Antonelli
//         Created:  Thu Jan 22 13:42:28CET 2009
// $Id: PhotonDataCertification.cc,v 1.3 2009/02/27 09:07:46 lantonel Exp $
//




using namespace std;

PhotonDataCertification::PhotonDataCertification(const edm::ParameterSet& pset)

{
  //cout << "entering constructor" << endl;
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  parameters_ = pset;
  //cout << "exiting constructor" << endl;

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
   
   

   //std::cout << "analyze" << std::endl;


}


// ------------ method called once each job just before starting event loop  ------------
void 
PhotonDataCertification::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PhotonDataCertification::endJob() {
}




// ------------ method called just before starting a new run  ------------
void 
PhotonDataCertification::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  
  //std::cout << ">>> BeginRun (PhotonDataCertification) <<<" << std::endl;
  //std::cout << ">>> run = " << run.id() << std::endl;

}

// ------------ method called right after a run ends ------------
void 
PhotonDataCertification::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  //std::cout << ">>> EndRun (PhotonDataCertification) <<<" << std::endl;
  //std::cout << ">>> run = " << run.id() << std::endl;
  //std::cout << "endJob" << std::endl;
  
  std::string filename    = parameters_.getUntrackedParameter<std::string>("fileName");
  std::string reffilename = parameters_.getUntrackedParameter<std::string>("refFileName");
  std::string outputFileName = parameters_.getUntrackedParameter<std::string>("outputFileName");

  dbe_->open(reffilename);

  dbe_->setCurrentFolder("/");

  std::vector<std::string> subDirVec;
  std::string RunDir;
  std::string RefRunDir;
  
  bool standalone = true;
  
  subDirVec = dbe_->getSubdirs();
   

  for (std::vector<std::string>::const_iterator ic = subDirVec.begin(); ic != subDirVec.end(); ic++) {
    
     //std::cout << "Dir = >>" << ic->c_str() << "<<" << std::endl;
    if (ic->find( "Run" ) != std::string::npos) {
      RefRunDir = *ic;
      //std::cout << "found RefRunDir" << std::endl;
    }
    if (ic->find( "Egamma" ) != std::string::npos) {
      standalone = false;
      //cout << " found Egamma" << endl;
    }
  }

  if(standalone) dbe_->setCurrentFolder(RefRunDir+"/Egamma/EventInfo/Certification/");  
  else dbe_->setCurrentFolder("Egamma/EventInfo/Certification/");

  MonitorElement* KolmogorovTest = dbe_->bookFloat("KolTest");
  MonitorElement* ChiSquaredTest = dbe_->bookFloat("Chi2Test");
  
  
  //std::cout << dbe_->pwd() << std::endl;
  //std::cout << "getting Histos" << std::endl;
  
  
  MonitorElement * RefHist = new MonitorElement(*(dbe_->get(RefRunDir+"/Egamma/Run summary/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtAllEcal")));
  //cout << "got first one " << endl;
  TH1F * RefHistTH1 = RefHist->getTH1F();
  
  MonitorElement * TestHist=0;
  if(standalone) TestHist = new MonitorElement(*(dbe_->get(RefRunDir+"/Egamma/Run summary/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtAllEcal")));
  else TestHist = new MonitorElement(*(dbe_->get("Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/phoEtAllEcal")));
  //cout << "got second one " << endl;
  TH1F * TestHistTH1 = TestHist->getTH1F();
  
  double KolTest;
  double Chi2Test;
  
  if(TestHistTH1->GetEntries() != 0){
 
    KolTest = TestHistTH1->KolmogorovTest(RefHistTH1);
    Chi2Test = TestHistTH1->Chi2Test(RefHistTH1);
    //cout << KolTest << "   " << Chi2Test << endl;
 
    KolmogorovTest->Fill(KolTest);
    ChiSquaredTest->Fill(Chi2Test);
  
  }
  else{
    KolmogorovTest->Fill(-1);
    ChiSquaredTest->Fill(-1);
  }

  if(standalone) dbe_->save(outputFileName);
  
  
}
