#include <iostream>



/**\class PhotonDataCertification 
*/
//
// Original Author: Jason Slaunwhite
//           
//         Created:  Thu Jan 22 13:42:28CET 2009
//

// system include files
#include <memory>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//root include files
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"



using namespace std;


//
// class decleration
//

class HLTOverallSummary : public edm::EDAnalyzer {

   public:
      explicit HLTOverallSummary(const edm::ParameterSet& pset);
      ~HLTOverallSummary();


      virtual void beginJob() override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) override ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) override ;



   private:

      DQMStore *dbe_;
      edm::ParameterSet parameters_;

      bool verbose_;

 // ----------member data ---------------------------
};



HLTOverallSummary::HLTOverallSummary(const edm::ParameterSet& pset)

{

  using namespace edm;
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    LogInfo ("HLTMuonVal") << "Can't find DQMStore, no results will be saved"
                           << endl;
  } else {
    dbe_->setVerbose(0);
  }
  
  parameters_ = pset;
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> Constructor (HLTOverallSummary) <<<" << endl;

}


HLTOverallSummary::~HLTOverallSummary()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTOverallSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if(verbose_) LogInfo ("HLTMuonVal")  << ">>> Analyze (HLTOverallSummary) <<<" << std::endl;

}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTOverallSummary::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTOverallSummary::endJob() 
{
}

// ------------ method called just before starting a new run  ------------
void 
HLTOverallSummary::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  using namespace edm;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> BeginRun (HLTOverallSummary) <<<" << std::endl;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> "<< run.id() << std::endl;

}

// ------------ method called right after a run ends ------------
void 
HLTOverallSummary::endRun(const edm::Run& run, const edm::EventSetup& c)
{

  using namespace edm;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> EndRun (HLTOverallSummary) <<<" << std::endl;

  if(!dbe_) {
    LogInfo ("HLTMuonVal") << "No dqmstore... skipping processing step" << endl;
    return;
  }
  
  std::vector<string> histoNameVector;

 
  //booking histograms according to naming conventions

  float defaultValueIfNotFound = 1.0;

  dbe_->setCurrentFolder("HLT/EventInfo/reportSummaryContent");


  //============ Unpack information ==========
  

  MonitorElement*  muonQualityBit = 0;
  muonQualityBit = dbe_->get("HLT_Muon");

  if (!muonQualityBit) {
    LogInfo ("HLTMuonVal") << "Can't find muonQuality bit... making a bit, setting it to zero" << endl;

    muonQualityBit = dbe_->bookFloat("HLT_Muon");
    muonQualityBit->Fill(defaultValueIfNotFound);
      
  }
                      
  MonitorElement*  eleQualityBit = 0;
  eleQualityBit = dbe_->get("HLT_Electron");

  if (!eleQualityBit) {
    LogInfo ("HLTMuonVal") << "Can't find eleQuality bit... making a bit, setting it to zero" << endl;

    eleQualityBit = dbe_->bookFloat("HLT_Electron");
    eleQualityBit->Fill(defaultValueIfNotFound);      
  }

  MonitorElement*  photonQualityBit = 0;
  photonQualityBit = dbe_->get("HLT_Photon");

  if (!photonQualityBit) {
    LogInfo ("HLTMuonVal") << "Can't find photonQuality bit... making a bit, setting it to zero" << endl;

    photonQualityBit = dbe_->bookFloat("HLT_Photon");
    photonQualityBit->Fill(defaultValueIfNotFound);      
  }


  //============ Book new storage locations =============

  dbe_->setCurrentFolder("HLT/EventInfo");
  MonitorElement*  hltQualityBit = dbe_->bookFloat("reportSummary");

  MonitorElement* hltQualitySummaryWord = dbe_->bookInt ("HLT_SUMMARY_WORD");

  //for now these will hold values from eta/phi tests for spikes/holes
  MonitorElement*  reportSummaryMap = dbe_->book2D("reportSummaryMap","HLT: ReportSummaryMap",3,-0.5,2.5,1,-0.5,0.5);
  MonitorElement*  CertificationSummaryMap = dbe_->book2D("certificationSummaryMap","HLT: CertificationSummaryMap",3,-0.5,2.5,1,-0.5,0.5);

  TH2 * reportSummaryMapTH2 = reportSummaryMap->getTH2F();

  reportSummaryMapTH2->GetXaxis()->SetBinLabel(1,"Muon");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(2,"Electron");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(3,"Photon");
  
  reportSummaryMapTH2->GetYaxis()->SetBinLabel(1,"Quality");


  TH2 * CertificationSummaryMapTH2 = CertificationSummaryMap->getTH2F();

  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(1,"Muon");    
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(2,"Electron");  
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(3,"Photon");                                                                          
  CertificationSummaryMapTH2->GetYaxis()->SetBinLabel(1,"Quality");



  //=================== Interpret bits and store result

  float photonValue = photonQualityBit->getFloatValue();
  
  float electronValue = eleQualityBit->getFloatValue();

  float muonValue = muonQualityBit->getFloatValue();

  float hltOverallValue = 1.0;
  
  if ( (photonValue > 0.99)
       && (electronValue > 0.99)
       && (muonValue > 0.99) ) {

    hltOverallValue = 1.0;

  } else {
    
    hltOverallValue = 0.0;
    
  }

  hltQualityBit->Fill(hltOverallValue);

  unsigned int hltSummaryValue = 0x0; //

  unsigned int ELECTRON_MASK = 0x1;
  unsigned int PHOTON_MASK = 0x2;  
  unsigned int MUON_MASK = 0x4;

  if (electronValue > 0.99) hltSummaryValue = hltSummaryValue | ELECTRON_MASK;    
  if (photonValue > 0.99) hltSummaryValue = hltSummaryValue | PHOTON_MASK;
  if (muonValue > 0.99) hltSummaryValue = hltSummaryValue | MUON_MASK;

  hltQualitySummaryWord->Fill(hltSummaryValue);

  reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(1,1), muonValue);
  reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(2,1), electronValue);
  reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(3,1), photonValue);

  CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(1,1), muonValue);
  CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(2,1), electronValue);
  CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(3,1), photonValue);

  


  
  
    
       
  

}


DEFINE_FWK_MODULE(HLTOverallSummary);
