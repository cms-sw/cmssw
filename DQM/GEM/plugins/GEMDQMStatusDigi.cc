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
  ~GEMDQMStatusDigi() override;
  
protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

private:
  unsigned int verbosity;
   
  edm::EDGetToken tagVFAT;
  edm::EDGetToken tagGEB;
  edm::EDGetToken tagAMC;

  int AMCBinN(uint16_t BID_);
  int GEBBinN(uint16_t BID_);
   
  MonitorElement *h1B1010All;
  MonitorElement *h1B1100All;
  MonitorElement *h1B1110All;
  
  MonitorElement *h1FlagAll;
  MonitorElement *h1CRCAll;
  
  MonitorElement *h1InputID;
  MonitorElement *h1Vwh;
  MonitorElement *h1Vwt;
  
  MonitorElement *h1GEBError;
  MonitorElement *h1GEBWarning;
  
  MonitorElement *h2B1010All;
  MonitorElement *h2B1100All;
  MonitorElement *h2B1110All;
  
  MonitorElement *h2FlagAll;
  MonitorElement *h2CRCAll;
  
  MonitorElement *h2InputID;
  MonitorElement *h2Vwh;
  MonitorElement *h2Vwt;
  
  MonitorElement *h2GEBError;
  MonitorElement *h2GEBWarning;

  MonitorElement *GEMDAV;  
  MonitorElement *Tstate;  
  MonitorElement *GDcount; 
  MonitorElement *ChamT;   
  MonitorElement *OOSG;    

  MonitorElement *GEMDAV2D;  
  MonitorElement *Tstate2D;  
  MonitorElement *GDcount2D; 
  MonitorElement *ChamT2D;   
  MonitorElement *OOSG2D;    
  
  std::unordered_map<uint16_t, int> mlAMCID; 
  std::unordered_map<uint16_t, int> mlGEBID; 

  
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------
GEMDQMStatusDigi::GEMDQMStatusDigi(const edm::ParameterSet& cfg)
{

  tagVFAT = consumes<GEMVfatStatusDigiCollection>(cfg.getParameter<edm::InputTag>("VFATInputLabel")); 
  tagGEB = consumes<GEMGEBStatusDigiCollection>(cfg.getParameter<edm::InputTag>("GEBInputLabel")); 
  tagAMC = consumes<GEMAMCStatusDigiCollection>(cfg.getParameter<edm::InputTag>("AMCInputLabel")); 
  
}

//----------------------------------------------------------------------------------------------------

GEMDQMStatusDigi::~GEMDQMStatusDigi()
{
}

//----------------------------------------------------------------------------------------------------


int GEMDQMStatusDigi::AMCBinN(uint16_t BID_) {
 
  if(mlAMCID.find(BID_) == mlAMCID.end()){
    int addIdx = mlAMCID.size();
    mlAMCID[BID_] = addIdx;
 
    int nNumAMC = (int)mlAMCID.size();
    string strLabel = "BID: "+to_string(BID_);
    const char* tmpLabel =  strLabel.data();
    ( (TH2F *)GEMDAV2D->getTH2F() )->SetBins(24, 0, 24, nNumAMC, 0, nNumAMC);
    ( (TH2F *)GEMDAV2D->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)Tstate2D->getTH2F() )->SetBins(15, 0, 15, nNumAMC, 0, nNumAMC);
    ( (TH2F *)Tstate2D->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)GDcount2D->getTH2F() )->SetBins(32, 0, 32, nNumAMC, 0, nNumAMC);
    ( (TH2F *)GDcount2D->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)ChamT2D->getTH2F() )->SetBins(24, 0, 24, nNumAMC, 0, nNumAMC);
    ( (TH2F *)ChamT2D->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  
    ( (TH2F *)OOSG2D->getTH2F() )->SetBins(1, 0, 1, nNumAMC, 0, nNumAMC);
    ( (TH2F *)OOSG2D->getTH2F() )->GetYaxis()->SetBinLabel(nNumAMC,tmpLabel);
  }
    
  return mlAMCID[BID_];
}

int GEMDQMStatusDigi::GEBBinN(uint16_t BID_){
  if(mlGEBID.find(BID_) == mlGEBID.end()){
    int addIdx = mlGEBID.size();
    mlGEBID[BID_] = addIdx;
    
    int nNumGEB = (int)mlGEBID.size();
    string strLabel = "BID: "+to_string(BID_);
    const char* tmpLabel =  strLabel.data();
    ( (TH2F *)h2B1010All->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1010All->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2B1100All->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1100All->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2B1110All->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2B1110All->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2FlagAll->getTH2F() )->SetBins(15, 0x0 , 0xf, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2FlagAll->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2CRCAll->getTH2F() )->SetBins(0xffff, -32768, 32768, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2CRCAll->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2InputID->getTH2F() )->SetBins(31,  0x0 , 0b11111, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2InputID->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2Vwh->getTH2F() )->SetBins(4095,  0x0 , 0xfff, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2Vwh->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2Vwt->getTH2F() )->SetBins(4095,  0x0 , 0xfff, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2Vwt->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
 
    ( (TH2F *)h2GEBError->getTH2F() )->SetBins(5, 0, 5, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2GEBError->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);

    ( (TH2F *)h2GEBWarning->getTH2F() )->SetBins(10,  0, 10, nNumGEB, 0, nNumGEB);
    ( (TH2F *)h2GEBWarning->getTH2F() )->GetYaxis()->SetBinLabel(nNumGEB,tmpLabel);
   
  }
  return mlGEBID[BID_];
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const & iSetup)
{

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/StatusDigi");
  
  h1B1010All = ibooker.book1D("vfatErrors_all_b1010", "Control Bit 1010", 15, 0x0 , 0xf);   
  h1B1100All = ibooker.book1D("vfatErrors_all_b1100", "Control Bit 1100", 15, 0x0 , 0xf);   
  h1B1110All = ibooker.book1D("vfatErrors_all_b1110", "Control Bit 1110", 15, 0x0 , 0xf);   
  h2B1010All = ibooker.book2D("vfatErrors_all_b1010_PerGEB", "Control Bit 1010", 15, 0x0 , 0xf, 1, 0, 1);   
  h2B1100All = ibooker.book2D("vfatErrors_all_b1100_PerGEB", "Control Bit 1100", 15, 0x0 , 0xf, 1, 0, 1);   
  h2B1110All = ibooker.book2D("vfatErrors_all_b1110_PerGEB", "Control Bit 1110", 15, 0x0 , 0xf, 1, 0, 1);   
  
  h1FlagAll = ibooker.book1D("vfatErrors_all_flag", "Control Flags", 15, 0x0 , 0xf);   
  h1CRCAll = ibooker.book1D("vfatErrors_all_CRC", "CRC Mismatches", 0xffff, -32768, 32768);   
  h2FlagAll = ibooker.book2D("vfatErrors_all_flag_PerGEB", "Control Flags", 15, 0x0 , 0xf, 1, 0, 1);   
  h2CRCAll = ibooker.book2D("vfatErrors_all_CRC_PerGEB", "CRC Mismatches", 0xffff, -32768, 32768, 1, 0, 1);   
  
  h1InputID = ibooker.book1D("GEB_InputID", "GEB GLIB input ID", 31,  0x0 , 0b11111);
  h1Vwh = ibooker.book1D("VFAT_Vwh", "VFAT word count", 4095,  0x0 , 0xfff);
  h1Vwt = ibooker.book1D("VFAT_Vwt", "VFAT word count", 4095,  0x0 , 0xfff);
  h2InputID = ibooker.book2D("GEB_InputID_PerGEB", "GEB GLIB input ID", 31,  0x0 , 0b11111, 1, 0, 1);
  h2Vwh = ibooker.book2D("VFAT_Vwh_PerGEB", "VFAT word count", 4095,  0x0 , 0xfff, 1, 0, 1);
  h2Vwt = ibooker.book2D("VFAT_Vwt_PerGEB", "VFAT word count", 4095,  0x0 , 0xfff, 1, 0, 1);
  
  h1GEBError = ibooker.book1D("GEB_Errors", "GEB Critical Errors", 5, 0, 5);
  h2GEBError = ibooker.book2D("GEB_Errors_PerGEB", "GEB Critical Errors", 5, 0, 5, 1, 0, 1);
  TH1F *histErr1D = h1GEBError->getTH1F();
  TH2F *histErr2D = h2GEBError->getTH2F();
  const char *error_flags[5] = {"Event Size Overflow", "L1AFIFO Full", "InFIFO Full", "Evt FIFO Full","InFIFO Underflow"};
  for (int i = 1; i<6; i++) {histErr1D->GetXaxis()->SetBinLabel(i, error_flags[i-1]); histErr2D->GetXaxis()->SetBinLabel(i, error_flags[i-1]);}
  h1GEBWarning = ibooker.book1D("GEB_Warnings", "GEB Warnings", 10,  0, 10);
  h2GEBWarning = ibooker.book2D("GEB_Warnings_PerGEB", "GEB Warnings", 10,  0, 10, 1, 0, 1);
  TH1F *histWar1D = h1GEBWarning->getTH1F();
  TH2F *histWar2D = h2GEBWarning->getTH2F();
  const char *warning_flags[10] = {"BX AMC-OH Mismatch", "BX AMC-VFAT Mismatch", "OOS AMC OH", "OOS AMC VFAT","No VFAT Marker","Event Size Warn", "L1AFIFO Near Full", "InFIFO Near Full", "EvtFIFO Near Full", "Stuck Data"};
  for (int i = 1; i<11; i++) {histWar1D->GetXaxis()->SetBinLabel(i, warning_flags[i-1]); histWar2D->GetXaxis()->SetBinLabel(i, warning_flags[i-1]);}
  
  GEMDAV = ibooker.book1D("GEMDAV", "GEM DAV list", 24,  0, 24);
  Tstate     = ibooker.book1D("Tstate", "TTS state", 15,  0, 15);
  GDcount    = ibooker.book1D("GDcount", "GEM DAV count", 32,  0, 32);
  ChamT      = ibooker.book1D("ChamT", "Chamber Timeout", 24, 0, 24);
  OOSG       = ibooker.book1D("OOSG", "OOS GLIB", 1, 0, 1);
  
  GEMDAV2D = ibooker.book2D("GEMDAV_PerAMC", "GEM DAV list", 24,  0, 24, 1, 0, 1);
  Tstate2D     = ibooker.book2D("Tstate_PerAMC", "TTS state", 15,  0, 15, 1, 0, 1);
  GDcount2D    = ibooker.book2D("GDcount_PerAMC", "GEM DAV count", 32,  0, 32, 1, 0, 1);
  ChamT2D      = ibooker.book2D("ChamT_PerAMC", "Chamber Timeout", 24, 0, 24, 1, 0, 1);
  OOSG2D       = ibooker.book2D("OOSG_PerAMC", "OOS GLIB", 1, 0, 1, 1, 0, 1);
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{

  edm::Handle<GEMVfatStatusDigiCollection> gemVFAT;
  edm::Handle<GEMGEBStatusDigiCollection> gemGEB;
  edm::Handle<GEMAMCStatusDigiCollection> gemAMC;
  event.getByToken( this->tagVFAT, gemVFAT);
  event.getByToken( this->tagGEB, gemGEB);
  event.getByToken( this->tagAMC, gemAMC);

  for (GEMVfatStatusDigiCollection::DigiRangeIterator vfatIt = gemVFAT->begin(); vfatIt != gemVFAT->end(); ++vfatIt){
    const GEMVfatStatusDigiCollection::Range& range = (*vfatIt).second;
    uint16_t tmpID = (*vfatIt).first;
    int nIdx = GEBBinN(tmpID);    
    for ( auto vfatError = range.first; vfatError != range.second; ++vfatError ) {
        
        h1B1010All->Fill(vfatError->getB1010());
        h1B1100All->Fill(vfatError->getB1100());
        h1B1110All->Fill(vfatError->getB1110());
        h1FlagAll->Fill(vfatError->getFlag());
        h1CRCAll->Fill(vfatError->getCrc());

        h2B1010All->Fill(vfatError->getB1010(), nIdx);
        h2B1100All->Fill(vfatError->getB1100(), nIdx);
        h2B1110All->Fill(vfatError->getB1110(), nIdx);
        h2FlagAll->Fill(vfatError->getFlag(), nIdx);
        h2CRCAll->Fill(vfatError->getCrc(), nIdx);
      }
    }

  for (GEMGEBStatusDigiCollection::DigiRangeIterator gebIt = gemGEB->begin(); gebIt != gemGEB->end(); ++gebIt){
    const GEMGEBStatusDigiCollection::Range& range = (*gebIt).second;
    for ( auto GEBStatus = range.first; GEBStatus != range.second; ++GEBStatus ) {
      uint16_t tmpID = (*gebIt).first;
      h1InputID->Fill(tmpID);
      h1Vwh->Fill(GEBStatus->getVwh());
      h1Vwt->Fill(GEBStatus->getVwt());

      int nIdx = GEBBinN(tmpID);
      h2InputID->Fill(tmpID, nIdx);
      h2Vwh->Fill(GEBStatus->getVwh(), nIdx);
      h2Vwt->Fill(GEBStatus->getVwt(), nIdx);
      
      for ( int bin = 0 ; bin < 9  ; bin++ ) {
        if ( ( ( GEBStatus->getErrorC() >> bin ) & 0x1 ) != 0 ) {
          h1GEBWarning->Fill(bin);
          h2GEBWarning->Fill(bin, nIdx);
        }
      }
      for ( int bin = 9 ; bin < 13 ; bin++ ) {
        if ( ( ( GEBStatus->getErrorC() >> bin ) & 0x1 ) != 0 ) {
          h1GEBError->Fill(bin - 9);
          h2GEBError->Fill(bin - 9, nIdx);
        }
      }
      
      if ( ( GEBStatus->getInFu()   & 0x1 ) != 0 ) {
        h1GEBError->Fill(9);
        h2GEBError->Fill(9, nIdx);
      }
      if ( ( GEBStatus->getStuckd() & 0x1 ) != 0 ) {
        h1GEBWarning->Fill(9);
        h2GEBWarning->Fill(9, nIdx);
      }
    }
  }

  for (GEMAMCStatusDigiCollection::DigiRangeIterator amcIt = gemAMC->begin(); amcIt != gemAMC->end(); ++amcIt){
    const GEMAMCStatusDigiCollection::Range& range = (*amcIt).second;
    for ( auto amc = range.first; amc != range.second; ++amc ) {
      uint16_t tmpID = (*amcIt).first;
      int nIdxAMC = AMCBinN(tmpID);
      uint8_t binFired = 0;
      for (int bin = 0; bin < 24; bin++){
        binFired = ((amc->GEMDAV() >> bin) & 0x1);
        if (binFired) {GEMDAV->Fill(bin); GEMDAV2D->Fill(bin, nIdxAMC);}
        binFired = ((amc->ChamT() >> bin) & 0x1);
        if (binFired) {ChamT->Fill(bin); ChamT2D->Fill(bin, nIdxAMC);}
      }
      
      Tstate->Fill(amc->Tstate());
      GDcount->Fill(amc->GDcount());
      OOSG->Fill(amc->OOSG());
      
      Tstate2D->Fill(amc->Tstate(), nIdxAMC);
      GDcount2D->Fill(amc->GDcount(), nIdxAMC);
      OOSG2D->Fill(amc->OOSG(), nIdxAMC);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//----------------------------------------------------------------------------------------------------

void GEMDQMStatusDigi::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(GEMDQMStatusDigi);
