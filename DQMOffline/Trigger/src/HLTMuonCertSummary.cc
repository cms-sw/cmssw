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
#include "DQMServices/Core/interface/DQMEDHarvester.h"

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

class HLTMuonCertSummary : public DQMEDHarvester {

   public:
      explicit HLTMuonCertSummary(const edm::ParameterSet& pset);
      ~HLTMuonCertSummary() override;


      void beginJob() override;
      void beginRun(const edm::Run&, const edm::EventSetup&) override ;
      void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override ;



   private:

      edm::ParameterSet parameters_;

      bool verbose_;

 // ----------member data ---------------------------
};



HLTMuonCertSummary::HLTMuonCertSummary(const edm::ParameterSet& pset)

{

  using namespace edm;

  parameters_ = pset;
  verbose_ = parameters_.getUntrackedParameter<bool>("verbose", false);
  
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> Constructor (HLTMuonCertSummary) <<<" << endl;

}



HLTMuonCertSummary::~HLTMuonCertSummary() = default;



//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
HLTMuonCertSummary::beginJob()
{

}



// ------------ method called just before starting a new run  ------------
void 
HLTMuonCertSummary::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  using namespace edm;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> BeginRun (HLTMuonCertSummary) <<<" << std::endl;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> "<< run.id() << std::endl;

}



// ------------ method called right after a run ends ------------
void 
HLTMuonCertSummary::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter)
{
  
  using namespace edm;
  if(verbose_) LogInfo ("HLTMuonVal")  << ">>> EndRun (HLTMuonCertSummary) <<<" << std::endl;
  
  std::vector<string> histoNameVector;

 
  //booking histograms according to naming conventions



  iBooker.setCurrentFolder("HLT/EventInfo/muonQuality");  

  MonitorElement*  reportSummary = iBooker.bookFloat("HLT_MUON_REPORT_SUMMARY");

  int SummaryBitResult = 100;
  
  MonitorElement*  CertificationSummary = iBooker.bookFloat("HLT_MUON_CERTIFICATION_SUMMARY");

  

  //for now these will hold values from eta/phi tests for spikes/holes
  MonitorElement*  reportSummaryMap = iBooker.book2D("HLT_MUON_ReportSummaryMap","HLT_MUON: ReportSummaryMap",6,-0.5,5.5,1,-0.5,0.5);
  MonitorElement*  CertificationSummaryMap = iBooker.book2D("HLT_MUON_CertificationSummaryMap","HLT_MUON: CertificationSummaryMap",6,-0.5,5.5,1,-0.5,0.5);

  TH2 * reportSummaryMapTH2 = reportSummaryMap->getTH2F();

  reportSummaryMapTH2->GetXaxis()->SetBinLabel(1,"PhiEtaOccAll");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(2,"PhiEtaOccMatch");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(3,"PhiEtaEff");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(4,"PhiEff");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(5,"EtaEff");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(6,"PtEff");
  
  reportSummaryMapTH2->GetYaxis()->SetBinLabel(1,"HLT_Mu5_allMuons");


  TH2 * CertificationSummaryMapTH2 = CertificationSummaryMap->getTH2F();

  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(1,"PhiEtaOccAll");    
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(2,"PhiEtaOccMatch");  
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(3,"PhiEtaEff");       
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(4,"PhiEff");          
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(5,"EtaEff");          
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(6,"PtEff");             
                                                                   
  CertificationSummaryMapTH2->GetYaxis()->SetBinLabel(1,"HLT_Mu5_allMuons");


  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recPhiVsRecEta_All");
  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recPhiVsRecEta_L3Filtered");
  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recEffPhiVsEta_L3Filtered");
  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recEffPt_L3Filtered");
  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recEffPhi_L3Filtered");
  histoNameVector.emplace_back("HLT/Muon/Distributions/HLT_Mu5/allMuons/recEffEta_L3Filtered");
  
  // to do:  what do we want in certification contents?
  //  iBooker.setCurrentFolder("Egamma/EventInfo/CertificationContents/"); 


  //   //looping over histograms to be tested
  if(verbose_) LogInfo ("HLTMuonVal")  << "\n>>> looping over histograms to be tested <<<\n\n";
  
  for(auto & it : histoNameVector){

    const string& HistoName = it;
    if(verbose_) LogInfo ("HLTMuonVal")  << ">>> " << HistoName;        
    

    MonitorElement * TestHist=nullptr;

    TestHist = iGetter.get(HistoName);

    bool validMe = TestHist!=nullptr;
    if(verbose_)  LogInfo ("HLTMuonVal")  << " is valid? " << validMe << "\n";
    if(!validMe) continue;

    string histNameNoPath =  TestHist->getName();
    
    if(verbose_)  LogInfo ("HLTMuonVal")  << ">>> TestHist Name: " << histNameNoPath << "\n\n";

  

    //get QReports associated to each ME
    std::vector<QReport *> myQReports = TestHist->getQReports();  
    if(verbose_)  LogInfo ("HLTMuonVal") << TestHist->getName() <<": myQReports.size() = " << myQReports.size() << "\n\n";
    for(uint i=0;i!=myQReports.size();++i) {
      
      std::string qtname = myQReports[i]->getQRName() ; // get QT name
      float qtresult = myQReports[i]->getQTresult(); // get QT result value
      int qtstatus = myQReports[i]->getStatus() ; // get QT status value 
      
      if(verbose_) LogInfo ("HLTMuonVal")  << "\tTest " << i << ":  qtname: " << qtname   << "\n";
      if(verbose_) LogInfo ("HLTMuonVal")  << "\tTest " << i << ":  qtresult: " << qtresult  << std::endl;
      if(verbose_) LogInfo ("HLTMuonVal")  << "\tTest " << i << ":  qtstatus: " << qtstatus    << "\n\n";

      
      //book and fill float for each test done
      iBooker.setCurrentFolder("HLT/EventInfo/muonQuality/");  
      MonitorElement * qValueInt  = iBooker.bookFloat(histNameNoPath+"_HLT_Mu5_"+qtname);
      qValueInt->Fill(qtstatus);

      // We're assuming that you want all of the bits to go into the decision


      if (HistoName.find("recPhiVsRecEta_All") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(1,1), qtstatus);
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(1,1), qtstatus );
        if ( (qtstatus == 200) && (SummaryBitResult < 300)) SummaryBitResult = 200;
        if ( qtstatus == 300 ) SummaryBitResult = 300;

      }

      if (HistoName.find("recPhiVsRecEta_L3Filtered") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(2,1), qtstatus);
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(2,1), qtstatus);
        if  ( (qtstatus == 200) && (SummaryBitResult < 300)) SummaryBitResult = 200;
        if (qtstatus == 300 ) SummaryBitResult = 300;

      }
      
      if (HistoName.find("recEffPhiVsEta_L3Filtered") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(3,1), qtstatus );
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(3,1), qtstatus);
        if ( (qtstatus == 200) && (SummaryBitResult < 300)) SummaryBitResult = 200;
        if (qtstatus == 300 ) SummaryBitResult = 300;
      }

      if (HistoName.find("recEffPhi_L3Filtered") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(4,1), qtstatus);
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(4,1), qtstatus);
      }

      if (HistoName.find("recEffEta_L3Filtered") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(5,1), qtstatus);
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(5,1), qtstatus);
      }
      
      if (HistoName.find("recEffPt_L3Filtered") != std::string::npos) {
        reportSummaryMapTH2->SetBinContent(reportSummaryMapTH2->GetBin(6,1), qtstatus);
        CertificationSummaryMapTH2->SetBinContent(CertificationSummaryMapTH2->GetBin(6,1), qtstatus);
      }
      

    }
  }


  reportSummary->Fill(SummaryBitResult);
  CertificationSummary->Fill(SummaryBitResult);



  // Set the final bits

  iBooker.setCurrentFolder("HLT/EventInfo/reportSummaryContents");
  MonitorElement* muonHLTQualityBinaryBit = iBooker.bookFloat ("HLT_Muon");
  
  if (SummaryBitResult == 100){
    muonHLTQualityBinaryBit->Fill(1);
  } else {
    muonHLTQualityBinaryBit->Fill(0);
  }  

}



DEFINE_FWK_MODULE(HLTMuonCertSummary);
