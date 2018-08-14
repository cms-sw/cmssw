#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GEMDigi/interface/GEMVfatStatusDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMGEBStatusDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCStatusDigiCollection.h"

#include <string>

//----------------------------------------------------------------------------------------------------
 
class GEMDQMStatusDigi: public DQMEDAnalyzer
{
public:
  GEMDQMStatusDigi(const edm::ParameterSet& cfg);
  ~GEMDQMStatusDigi() override {};
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions); 

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override {};
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override {};
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override {};

private:
  int nVfat_ = 24;
  int cBit_ = 9;
  int eBit_ = 13;
  edm::EDGetToken tagVFAT_;
  edm::EDGetToken tagGEB_;
  edm::EDGetToken tagAMC_;

  int AMCBinN(uint16_t BID_);
  int GEBBinN(uint16_t BID_);
   
  MonitorElement *h1B1010All_;
  MonitorElement *h1B1100All_;
  MonitorElement *h1B1110All_;
  
  MonitorElement *h1FlagAll_;
  MonitorElement *h1CRCAll_;
  
  MonitorElement *h1InputID_;
  MonitorElement *h1Vwh_;
  MonitorElement *h1Vwt_;
  
  MonitorElement *h1GEBError_;
  MonitorElement *h1GEBWarning_;
  
  MonitorElement *h2B1010All_;
  MonitorElement *h2B1100All_;
  MonitorElement *h2B1110All_;
  
  MonitorElement *h2FlagAll_;
  MonitorElement *h2CRCAll_;
  
  MonitorElement *h2InputID_;
  MonitorElement *h2Vwh_;
  MonitorElement *h2Vwt_;
  
  MonitorElement *h2GEBError_;
  MonitorElement *h2GEBWarning_;

  MonitorElement *GEMDAV_;  
  MonitorElement *Tstate_;  
  MonitorElement *GDcount_; 
  MonitorElement *ChamT_;   
  MonitorElement *OOSG_;    

  MonitorElement *GEMDAV2D_;  
  MonitorElement *Tstate2D_;  
  MonitorElement *GDcount2D_; 
  MonitorElement *ChamT2D_;   
  MonitorElement *OOSG2D_;    
  
  std::unordered_map<uint16_t, int> mlAMCID_; 
  std::unordered_map<uint16_t, int> mlGEBID_; 

  
};

using namespace std;
using namespace edm;

GEMDQMStatusDigi::GEMDQMStatusDigi(const edm::ParameterSet& cfg)
{

  tagVFAT_ = consumes<GEMVfatStatusDigiCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel")); 
  tagGEB_ = consumes<GEMGEBStatusDigiCollection>(cfg.getParameter<edm::InputTag>("GEBInputLabel")); 
  tagAMC_ = consumes<GEMAMCStatusDigiCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel")); 
  
}

void GEMDQMStatusDigi::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("VFATInputLabel", edm::InputTag("muonGEMDigis", "vfatStatus")); 
  desc.add<edm::InputTag>("GEBInputLabel", edm::InputTag("muonGEMDigis", "GEBStatus")); 
  desc.add<edm::InputTag>("AMCInputLabel", edm::InputTag("muonGEMDigis", "AMCStatus")); 
  descriptions.add("GEMDQMStatusDigi", desc);  
}

int GEMDQMStatusDigi::AMCBinN(uint16_t BID_) {
 
  if(mlAMCID_.find(BID_) == mlAMCID_.end()){
    int addIdx = mlAMCID_.size();
    mlAMCID_[BID_] = addIdx;
 
    int nNumAMC = (int)mlAMCID_.size();
    string strLabel = "BID: "+to_string(BID_);
    const char* tmpLabel =  strLabel.data();
    ( (TH2F *)GEMDAV2D_->getTH2F() )->SetBins(24, 0, 24, nNumAMC, 0, nNumAMC);
    ( (TH2F *)GEMDAV2D_->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)Tstate2D_->getTH2F() )->SetBins(15, 0, 15, nNumAMC, 0, nNumAMC);
    ( (TH2F *)Tstate2D_->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)GDcount2D_->getTH2F() )->SetBins(32, 0, 32, nNumAMC, 0, nNumAMC);
    ( (TH2F *)GDcount2D_->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)ChamT2D_->getTH2F() )->SetBins(24, 0, 24, nNumAMC, 0, nNumAMC);
    ( (TH2F *)ChamT2D_->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)OOSG2D_->getTH2F() )->SetBins(1, 0, 1, nNumAMC, 0, nNumAMC);
    ( (TH2F *)OOSG2D_->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  }
    
  return mlAMCID_[BID_];
}

int GEMDQMStatusDigi::GEBBinN(uint16_t BID_){
  if(mlGEBID_.find(BID_) == mlGEBID_.end()){
    int addIdx = mlGEBID_.size();
    mlGEBID_[BID_] = addIdx;
    
    int nNumGEB = (int)mlGEBID_.size();
    string strLabel = "BID: "+to_string(BID_);
    const char* tmpLabel =  strLabel.data();
    ( (TH2F *)h2B1010All_->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1010All_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2B1100All_->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1100All_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2B1110All_->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1110All_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2FlagAll_->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2FlagAll_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2CRCAll_->getTH2F() )->SetBins(0xffff, -32768, 32768, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2CRCAll_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2InputID_->getTH2F() )->SetBins(31,  0x0 , 0b11111, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2InputID_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2Vwh_->getTH2F() )->SetBins(4095,  0x0 , 0xfff, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2Vwh_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2Vwt_->getTH2F() )->SetBins(4095,  0x0 , 0xfff, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2Vwt_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2GEBError_->getTH2F() )->SetBins(5, 0, 5, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2GEBError_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2GEBWarning_->getTH2F() )->SetBins(10,  0, 10, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2GEBWarning_->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
   
  }
  return mlGEBID_[BID_];
}

//----------------------------------------------------------------------------------------------------


void GEMDQMStatusDigi::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const & iSetup)
{

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/StatusDigi");
  
  h1B1010All_ = ibooker.book1D("vfatErrors_all_b1010", "Control Bit 1010", 15, 0x0 , 0xf);   
  h1B1100All_ = ibooker.book1D("vfatErrors_all_b1100", "Control Bit 1100", 15, 0x0 , 0xf);   
  h1B1110All_ = ibooker.book1D("vfatErrors_all_b1110", "Control Bit 1110", 15, 0x0 , 0xf);   
  h2B1010All_ = ibooker.book2D("vfatErrors_all_b1010_PerGEB", "Control Bit 1010", 15, 0x0 , 0xf, 1, 0, 1);   
  h2B1100All_ = ibooker.book2D("vfatErrors_all_b1100_PerGEB", "Control Bit 1100", 15, 0x0 , 0xf, 1, 0, 1);   
  h2B1110All_ = ibooker.book2D("vfatErrors_all_b1110_PerGEB", "Control Bit 1110", 15, 0x0 , 0xf, 1, 0, 1);   
  
  h1FlagAll_ = ibooker.book1D("vfatErrors_all_flag", "Control Flags", 15, 0x0 , 0xf);   
  h1CRCAll_ = ibooker.book1D("vfatErrors_all_CRC", "CRC Mismatches", 0xffff, -32768, 32768);   
  h2FlagAll_ = ibooker.book2D("vfatErrors_all_flag_PerGEB", "Control Flags", 15, 0x0 , 0xf, 1, 0, 1);   
  h2CRCAll_ = ibooker.book2D("vfatErrors_all_CRC_PerGEB", "CRC Mismatches", 0xffff, -32768, 32768, 1, 0, 1);   
  
  h1InputID_ = ibooker.book1D("GEB_InputID", "GEB GLIB input ID", 31,  0x0 , 0b11111);
  h1Vwh_ = ibooker.book1D("VFAT_Vwh", "VFAT word count", 4095,  0x0 , 0xfff);
  h1Vwt_ = ibooker.book1D("VFAT_Vwt", "VFAT word count", 4095,  0x0 , 0xfff);
  h2InputID_ = ibooker.book2D("GEB_InputID_PerGEB", "GEB GLIB input ID", 31,  0x0 , 0b11111, 1, 0, 1);
  h2Vwh_ = ibooker.book2D("VFAT_Vwh_PerGEB", "VFAT word count", 4095,  0x0 , 0xfff, 1, 0, 1);
  h2Vwt_ = ibooker.book2D("VFAT_Vwt_PerGEB", "VFAT word count", 4095,  0x0 , 0xfff, 1, 0, 1);
  
  h1GEBError_ = ibooker.book1D("GEB_Errors", "GEB Critical Errors", 5, 0, 5);
  h2GEBError_ = ibooker.book2D("GEB_Errors_PerGEB", "GEB Critical Errors", 5, 0, 5, 1, 0, 1);
  TH1F *histErr1D = h1GEBError_->getTH1F();
  TH2F *histErr2D = h2GEBError_->getTH2F();
  const char *error_flags[5] = {"Event Size Overflow", "L1AFIFO Full", "InFIFO Full", "Evt FIFO Full","InFIFO Underflow"};
  for (int i = 1; i< histErr1D->GetNbinsX()+1; i++) {histErr1D->GetXaxis()->SetBinLabel(i, error_flags[i-1]); histErr2D->GetXaxis()->SetBinLabel(i, error_flags[i-1]);}
  h1GEBWarning_ = ibooker.book1D("GEB_Warnings", "GEB Warnings", 10,  0, 10);
  h2GEBWarning_ = ibooker.book2D("GEB_Warnings_PerGEB", "GEB Warnings", 10,  0, 10, 1, 0, 1);
  TH1F *histWar1D = h1GEBWarning_->getTH1F();
  TH2F *histWar2D = h2GEBWarning_->getTH2F();
  const char *warning_flags[10] = {"BX AMC-OH Mismatch", "BX AMC-VFAT Mismatch", "OOS AMC OH", "OOS AMC VFAT","No VFAT Marker","Event Size Warn", "L1AFIFO Near Full", "InFIFO Near Full", "EvtFIFO Near Full", "Stuck Data"};
  for (int i = 1; i<histWar1D->GetNbinsX()+1; i++) {histWar1D->GetXaxis()->SetBinLabel(i, warning_flags[i-1]); histWar2D->GetXaxis()->SetBinLabel(i, warning_flags[i-1]);}
  
  GEMDAV_ = ibooker.book1D("GEMDAV", "GEM DAV list", 24,  0, 24);
  Tstate_     = ibooker.book1D("Tstate", "TTS state", 15,  0, 15);
  GDcount_    = ibooker.book1D("GDcount", "GEM DAV count", 32,  0, 32);
  ChamT_      = ibooker.book1D("ChamT", "Chamber Timeout", 24, 0, 24);
  OOSG_       = ibooker.book1D("OOSG", "OOS GLIB", 1, 0, 1);
  
  GEMDAV2D_ = ibooker.book2D("GEMDAV_PerAMC", "GEM DAV list", 24,  0, 24, 1, 0, 1);
  Tstate2D_     = ibooker.book2D("Tstate_PerAMC", "TTS state", 15,  0, 15, 1, 0, 1);
  GDcount2D_    = ibooker.book2D("GDcount_PerAMC", "GEM DAV count", 32,  0, 32, 1, 0, 1);
  ChamT2D_      = ibooker.book2D("ChamT_PerAMC", "Chamber Timeout", 24, 0, 24, 1, 0, 1);
  OOSG2D_       = ibooker.book2D("OOSG_PerAMC", "OOS GLIB", 1, 0, 1, 1, 0, 1);
}

void GEMDQMStatusDigi::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  edm::Handle<GEMVfatStatusDigiCollection> gemVFAT;
  edm::Handle<GEMGEBStatusDigiCollection> gemGEB;
  edm::Handle<GEMAMCStatusDigiCollection> gemAMC;
  event.getByToken( this->tagVFAT_, gemVFAT);
  event.getByToken( this->tagGEB_, gemGEB);
  event.getByToken( this->tagAMC_, gemAMC);

  for (GEMVfatStatusDigiCollection::DigiRangeIterator vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt){
    const GEMVfatStatusDigiCollection::Range& range = (*vfatIt).second;
    uint16_t tmpID = (*vfatIt).first;
    int nIdx = GEBBinN(tmpID);    
    for ( auto vfatError = range.first; vfatError != range.second; ++vfatError ) {
        
        h1B1010All_->Fill(vfatError->getB1010());
        h1B1100All_->Fill(vfatError->getB1100());
        h1B1110All_->Fill(vfatError->getB1110());
        h1FlagAll_->Fill(vfatError->getFlag());
        h1CRCAll_->Fill(vfatError->getCrc());

        h2B1010All_->Fill(vfatError->getB1010(), nIdx);
        h2B1100All_->Fill(vfatError->getB1100(), nIdx);
        h2B1110All_->Fill(vfatError->getB1110(), nIdx);
        h2FlagAll_->Fill(vfatError->getFlag(), nIdx);
        h2CRCAll_->Fill(vfatError->getCrc(), nIdx);
      }
    }

  for (GEMGEBStatusDigiCollection::DigiRangeIterator gebIt = gemGEB->begin(); gebIt != gemGEB->end(); ++gebIt){
    const GEMGEBStatusDigiCollection::Range& range = (*gebIt).second;
    for ( auto GEBStatus = range.first; GEBStatus != range.second; ++GEBStatus ) {
      uint16_t tmpID = (*gebIt).first;
      h1InputID_->Fill(tmpID);
      h1Vwh_->Fill(GEBStatus->getVwh());
      h1Vwt_->Fill(GEBStatus->getVwt());

      int nIdx = GEBBinN(tmpID);
      h2InputID_->Fill(tmpID, nIdx);
      h2Vwh_->Fill(GEBStatus->getVwh(), nIdx);
      h2Vwt_->Fill(GEBStatus->getVwt(), nIdx);
      
      for ( int bin = 0 ; bin < cBit_  ; bin++ ) {
        if ( ( ( GEBStatus->getErrorC() >> bin ) & 0x1 ) != 0 ) {
          h1GEBWarning_->Fill(bin);
          h2GEBWarning_->Fill(bin, nIdx);
        }
      }
      for ( int bin = cBit_ ; bin < eBit_ ; bin++ ) {
        if ( ( ( GEBStatus->getErrorC() >> bin ) & 0x1 ) != 0 ) {
          h1GEBError_->Fill(bin - 9);
          h2GEBError_->Fill(bin - 9, nIdx);
        }
      }
      
      if ( ( GEBStatus->getInFu()   & 0x1 ) != 0 ) {
        h1GEBError_->Fill(9);
        h2GEBError_->Fill(9, nIdx);
      }
      if ( ( GEBStatus->getStuckd() & 0x1 ) != 0 ) {
        h1GEBWarning_->Fill(9);
        h2GEBWarning_->Fill(9, nIdx);
      }
    }
  }

  for (GEMAMCStatusDigiCollection::DigiRangeIterator amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt){
    const GEMAMCStatusDigiCollection::Range& range = (*amcIt).second;
    for ( auto amc = range.first; amc != range.second; ++amc ) {
      uint16_t tmpID = (*amcIt).first;
      int nIdxAMC = AMCBinN(tmpID);
      uint8_t binFired = 0;
      for (int bin = 0; bin < nVfat_; bin++){
        binFired = ((amc->GEMDAV() >> bin) & 0x1);
        if (binFired) {GEMDAV_->Fill(bin); GEMDAV2D_->Fill(bin, nIdxAMC);}
        binFired = ((amc->ChamT() >> bin) & 0x1);
        if (binFired) {ChamT_->Fill(bin); ChamT2D_->Fill(bin, nIdxAMC);}
      }
      
      Tstate_->Fill(amc->Tstate());
      GDcount_->Fill(amc->GDcount());
      OOSG_->Fill(amc->OOSG());
      
      Tstate2D_->Fill(amc->Tstate(), nIdxAMC);
      GDcount2D_->Fill(amc->GDcount(), nIdxAMC);
      OOSG2D_->Fill(amc->OOSG(), nIdxAMC);
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMStatusDigi);
