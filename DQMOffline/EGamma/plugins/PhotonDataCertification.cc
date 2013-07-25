#include <iostream>

#include "DQMOffline/EGamma/plugins/PhotonDataCertification.h"
#include "FWCore/Framework/interface/Run.h"

/**\class PhotonDataCertification
*/
//
// Original Author:  Louis James Antonelli
//         Created:  Thu Jan 22 13:42:28CET 2009
// $Id: PhotonDataCertification.cc,v 1.1 2011/04/08 15:55:00 chamont Exp $
//




using namespace std;

PhotonDataCertification::PhotonDataCertification(const edm::ParameterSet& pset)

{

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  parameters_ = pset;
  verbose_ = parameters_.getParameter<bool>("verbose");

  if(verbose_) cout << ">>> Constructor (PhotonDataCertification) <<<" << endl;

}


PhotonDataCertification::~PhotonDataCertification()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PhotonDataCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   if(verbose_) std::cout << ">>> Analyze (PhotonDataCertification) <<<" << std::endl;

}

// ------------ method called once each job just before starting event loop  ------------
void
PhotonDataCertification::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
PhotonDataCertification::endJob()
{
}

// ------------ method called just before starting a new run  ------------
void
PhotonDataCertification::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  if(verbose_) std::cout << ">>> BeginRun (PhotonDataCertification) <<<" << std::endl;
  if(verbose_) std::cout << ">>> "<< run.id() << std::endl;

}

// ------------ method called right after a run ends ------------
void
PhotonDataCertification::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if(verbose_) std::cout << ">>> EndRun (PhotonDataCertification) <<<" << std::endl;

  std::vector<string> histoNameVector;


  //booking histograms according to naming conventions

  if(!dbe_) return;

  dbe_->setCurrentFolder("Egamma/EventInfo/");

  MonitorElement*  reportSummary = dbe_->bookFloat("reportSummary");
  MonitorElement*  CertificationSummary = dbe_->bookFloat("CertificationSummary");

  //for now these will hold values from eta/phi tests for spikes/holes
  MonitorElement*  reportSummaryMap = dbe_->book2D("reportSummaryMap","reportSummaryMap",2,0,2,2,0,2);
  MonitorElement*  CertificationSummaryMap = dbe_->book2D("CertificationSummaryMap","CertificationSummaryMap",2,0,2,2,0,2);

  TH2 * reportSummaryMapTH2 = reportSummaryMap->getTH2F();

  reportSummaryMapTH2->GetXaxis()->SetBinLabel(1,"Eta");
  reportSummaryMapTH2->GetXaxis()->SetBinLabel(2,"Phi");
  reportSummaryMapTH2->GetYaxis()->SetBinLabel(1,"SpikeTest");
  reportSummaryMapTH2->GetYaxis()->SetBinLabel(2,"HoleTest");

  TH2 * CertificationSummaryMapTH2 = CertificationSummaryMap->getTH2F();

  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(1,"Eta");
  CertificationSummaryMapTH2->GetXaxis()->SetBinLabel(2,"Phi");
  CertificationSummaryMapTH2->GetYaxis()->SetBinLabel(1,"SpikeTest");
  CertificationSummaryMapTH2->GetYaxis()->SetBinLabel(2,"HoleTest");



  //histograms to be tested and results put into Egamma/EventInfo/reportSummaryContents
  histoNameVector.push_back("nPhoAllEcal");
  histoNameVector.push_back("r9AllEcal");
  histoNameVector.push_back("phoEta");
  histoNameVector.push_back("phoPhi");


//   histoNameVector.push_back("phoEtAllEcal");
//   histoNameVector.push_back("EfficiencyVsEtHLT");
//   histoNameVector.push_back("EfficiencyVsEtaHLT");
//   histoNameVector.push_back("EfficiencyVsEtLoose");
//   histoNameVector.push_back("EfficiencyVsEtaLoose");


  // to do:  what do we want in certification contents?
  //  dbe_->setCurrentFolder("Egamma/EventInfo/CertificationContents/");


  //looping over histograms to be tested
  if(verbose_) std::cout << "\n>>> looping over histograms to be tested <<<\n\n";
  for(std::vector<string>::iterator it=histoNameVector.begin();it!=histoNameVector.end();++it){

    string HistoName = (*it);
    if(verbose_) std::cout << ">>> " << HistoName;

    MonitorElement * TestHist=0;

    if(HistoName.find("Efficiency")!=std::string::npos){
      TestHist = dbe_->get("Egamma/PhotonAnalyzer/Efficiencies/"+HistoName);
    }
    else{
      TestHist = dbe_->get("Egamma/PhotonAnalyzer/GoodCandidatePhotons/Et above 0 GeV/"+HistoName);
    }
    bool validMe = TestHist!=0;
    if(verbose_)  std::cout << " is valid? " << validMe << "\n";
    if(!validMe) continue;

    if(verbose_)  std::cout << ">>> TestHist Name: " << TestHist->getName() << "\n\n";



    //get QReports associated to each ME
    std::vector<QReport *> myQReports = TestHist->getQReports();
    if(verbose_)  cout << TestHist->getName() <<": myQReports.size() = " << myQReports.size() << "\n\n";
    for(uint i=0;i!=myQReports.size();++i) {

      std::string qtname = myQReports[i]->getQRName() ; // get QT name
      float qtresult = myQReports[i]->getQTresult(); // get QT result value
      int qtstatus = myQReports[i]->getStatus() ; // get QT status value

      if(verbose_) std::cout << "\tTest " << i << ":  qtname: " << qtname   << "\n";
      if(verbose_) std::cout << "\tTest " << i << ":  qtresult: " << qtresult  << std::endl;
      if(verbose_) std::cout << "\tTest " << i << ":  qtstatus: " << qtstatus    << "\n\n";


      //book and fill float for each test done
      dbe_->setCurrentFolder("Egamma/EventInfo/reportSummaryContents/");
      MonitorElement * Float  = dbe_->bookFloat(HistoName+"_"+qtname);
      Float->Fill(qtresult);

    }
  }



  //filling summaries based on geometric spikes/holes

  float etaSpikeTestResult=0;
  float phiSpikeTestResult=0;
  float etaHoleTestResult=0;
  float phiHoleTestResult=0;

  if(dbe_->get("Egamma/EventInfo/reportSummaryContents/phoEta_SpikeTest")!=0)
    etaSpikeTestResult = dbe_->get("Egamma/EventInfo/reportSummaryContents/phoEta_SpikeTest")->getFloatValue();
  else  etaSpikeTestResult = -1;

  if(dbe_->get("Egamma/EventInfo/reportSummaryContents/phoPhi_SpikeTest")!=0)
    phiSpikeTestResult = dbe_->get("Egamma/EventInfo/reportSummaryContents/phoPhi_SpikeTest")->getFloatValue();
  else  phiSpikeTestResult = -1;

  if(dbe_->get("Egamma/EventInfo/reportSummaryContents/phoEta_HoleTest")!=0)
    etaHoleTestResult = dbe_->get("Egamma/EventInfo/reportSummaryContents/phoEta_HoleTest")->getFloatValue();
  else  etaHoleTestResult = -1;

  if(dbe_->get("Egamma/EventInfo/reportSummaryContents/phoPhi_HoleTest")!=0)
    phiHoleTestResult = dbe_->get("Egamma/EventInfo/reportSummaryContents/phoPhi_HoleTest")->getFloatValue();
  else  phiHoleTestResult = -1;



  if(verbose_) std::cout << ">>>  Results of tests to be put into Summary Maps  <<<\n\n";
  if(verbose_) std::cout << "\tetaSpikeTestResult= " << etaSpikeTestResult << "\n";
  if(verbose_) std::cout << "\tphiSpikeTestResult= " << phiSpikeTestResult << "\n";
  if(verbose_) std::cout << "\tetaHoleTestResult= " << etaHoleTestResult << "\n";
  if(verbose_) std::cout << "\tphiHoleTestResult= " << phiHoleTestResult << "\n\n";


  //fill reportSummary & CertificationSummary with average of hole/spike tests
  float reportSummaryFloat = (etaSpikeTestResult+etaHoleTestResult+phiSpikeTestResult+phiHoleTestResult)/4.;
  if(reportSummaryFloat<0) reportSummaryFloat = -1;
  reportSummary->Fill(reportSummaryFloat);
  CertificationSummary->Fill(reportSummaryFloat);

  reportSummaryMap->Fill(0,0,etaSpikeTestResult);
  reportSummaryMap->Fill(0,1,etaHoleTestResult);
  reportSummaryMap->Fill(1,0,phiSpikeTestResult);
  reportSummaryMap->Fill(1,1,phiHoleTestResult);

  CertificationSummaryMap->Fill(0,0,etaSpikeTestResult);
  CertificationSummaryMap->Fill(0,1,etaHoleTestResult);
  CertificationSummaryMap->Fill(1,0,phiSpikeTestResult);
  CertificationSummaryMap->Fill(1,1,phiHoleTestResult);




}
