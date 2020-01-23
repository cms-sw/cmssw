#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <TH2F.h>
#include <TFile.h>
#include <TDirectoryFile.h>
#include <TKey.h>


using namespace edm;


class GEMDQMHarvester: public DQMEDHarvester
{  
public:
  GEMDQMHarvester(const edm::ParameterSet&);
  ~GEMDQMHarvester() override {};
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
protected:
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; // Cannot use; it is called after dqmSaver
  
  void refineTimeHistograms(edm::Service<DQMStore> &);
  void refineTimeHistograms(DQMStore::IBooker &, DQMStore::IGetter &);
  void refineTimeHistograms(TFile *);
  void refineTimeHistogramsCore(TH2F *, std::string &, TH2F *&, std::string strTmpPrefix = "tmp_");
  
  void createSummaryPlots(edm::Service<DQMStore> &);
  
  std::string strOutFile_;
};


GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet &cfg) {
  strOutFile_ = cfg.getParameter<std::string>("fromFile");
}


void GEMDQMHarvester::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>("fromFile", "");
  descriptions.add("GEMDQMHarvester", desc);  
}


void GEMDQMHarvester::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  return;
  if ( strOutFile_.empty() ) {
    refineTimeHistograms(ibooker, igetter);
  } else {
    TFile *fDQM = TFile::Open(strOutFile_.c_str(), "UPDATE");
    
    refineTimeHistograms(fDQM);
    
    fDQM->Close();
  }
}


void GEMDQMHarvester::endRun(edm::Run const&, edm::EventSetup const&) {
  /*edm::Service<DQMStore> store;
  
  refineTimeHistograms(store);
  createSummaryPlots(store);*/
}


void GEMDQMHarvester::refineTimeHistograms(edm::Service<DQMStore> &store) {
  store->setCurrentFolder("GEM/StatusDigi");
  auto listME = store->getMEs();
  
  for ( auto strName : listME ) {
    if ( strName.find("primitive_per_time_") == std::string::npos ) continue;
    
    MonitorElement *h2Curr = store->get("GEM/StatusDigi/" + strName);
    std::string strNewName;
    TH2F *h2New;
    
    refineTimeHistogramsCore(h2Curr->getTH2F(), strNewName, h2New);
    
    store->book2D(strNewName, h2New);
    store->removeElement(strName);
  }
}


void GEMDQMHarvester::refineTimeHistograms(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.cd("GEM/StatusDigi");
  ibooker.setCurrentFolder("GEM/StatusDigi");
  auto listME = igetter.getMEs();
  
  for ( auto strName : listME ) {
    if ( strName.find("primitive_per_time_") == std::string::npos ) continue;
    
    MonitorElement *h2Curr = igetter.get("GEM/StatusDigi/" + strName);
    std::string strNewName;
    TH2F *h2New;
    
    refineTimeHistogramsCore(h2Curr->getTH2F(), strNewName, h2New);
    
    ibooker.book2D(strNewName, h2New);
    igetter.removeElement(strName);
  }
}


void GEMDQMHarvester::refineTimeHistograms(TFile *fDQM) {
  std::string strRunnum = ( (TDirectoryFile *)fDQM->Get("DQMData") )->GetListOfKeys()->At(0)->GetName();
  std::string strPathMain = "DQMData/" + strRunnum + "/GEM/Run summary/StatusDigi";
  
  fDQM->cd(strPathMain.c_str());
  auto dirMain = ( (TDirectoryFile *)fDQM->Get(strPathMain.c_str()) );
  
  for ( auto k : *dirMain->GetListOfKeys() ) {
    TKey *key = (TKey *)k;
    std::string strName = key->ReadObj()->GetName();
    std::string strNewName;
    
    if ( strName.find("primitive_per_time_") == std::string::npos ) continue;
    
    TH2F *h2Curr = (TH2F *)fDQM->Get(( strPathMain + "/" + strName ).c_str());
    TH2F *h2New;
    refineTimeHistogramsCore(h2Curr, strNewName, h2New, "");
    
    fDQM->Write();
  }
  
  for ( auto k : *dirMain->GetListOfKeys() ) std::cout << ( (TKey *)k )->ReadObj()->GetName() << std::endl;
}


void GEMDQMHarvester::refineTimeHistogramsCore(TH2F *h2Curr, std::string &strNewName, 
                                               TH2F *&h2New, std::string strTmpPrefix)
{
  strNewName = std::string(h2Curr->GetName()).substr(std::string("primitive_").size());
  
  Int_t nNbinsX = h2Curr->GetNbinsX();
  Int_t nNbinsY = h2Curr->GetNbinsY();
  Int_t nNbinsXActual = 0;
  
  for ( nNbinsXActual = 0 ; nNbinsXActual < nNbinsX ; nNbinsXActual++ ) {
    if ( h2Curr->GetBinContent(nNbinsXActual + 1, 0) <= 0 ) break;
  }
  
  std::string strTitle = std::string(h2Curr->GetTitle()) + ";" + 
    std::string(h2Curr->GetXaxis()->GetTitle()) + ";" + std::string(h2Curr->GetYaxis()->GetTitle());
  std::cout << strTitle << "(" << nNbinsXActual << ", " << nNbinsY << ")" << std::endl;
  
  strTitle = "";
  h2New = new TH2F(( strTmpPrefix + strNewName ).c_str(), strTitle.c_str(), 
      nNbinsXActual, 0.0, (Double_t)nNbinsXActual, nNbinsY, 0.0, (Double_t)nNbinsY);
  
  for ( Int_t i = 1 ; false && i <= nNbinsY ; i++ ) {
    std::string strLabel = h2Curr->GetYaxis()->GetBinLabel(i);
    if ( !strLabel.empty() ) h2New->GetYaxis()->SetBinLabel(i, strLabel.c_str());
  }
  
  for ( Int_t j = 0 ; false && j <= nNbinsY ; j++ ) 
  for ( Int_t i = 1 ; i <= nNbinsXActual ; i++ ) {
    h2New->SetBinContent(i, j, h2Curr->GetBinContent(i, j));
  }
}


void GEMDQMHarvester::createSummaryPlots(edm::Service<DQMStore> &store) {
}


DEFINE_FWK_MODULE(GEMDQMHarvester);
