#include "DQM/HLTEvF/interface/HLTMonMuonClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TRandom.h"
using namespace edm;
using namespace std;

HLTMonMuonClient::HLTMonMuonClient(const edm::ParameterSet& ps){
 
  indir_   = ps.getUntrackedParameter<string>("input_dir","HLT/HLTMonMuon/Summary");
  outdir_  = ps.getUntrackedParameter<string>("output_dir","HLT/HLTMonMuon/Client");    

  dbe_ = NULL;
  //if (ps.getUntrackedParameter < bool > ("DQMStore", false)) {
  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);
    //}

  if (dbe_ != NULL) {
    dbe_->setCurrentFolder(outdir_);
  }

}

HLTMonMuonClient::~HLTMonMuonClient(){}

//--------------------------------------------------------
void HLTMonMuonClient::beginJob(const EventSetup& context){
  
}

//--------------------------------------------------------
void HLTMonMuonClient::beginRun(const Run& r, const EventSetup& context) {
 
  if (dbe_) {
    dbe_->setCurrentFolder(indir_+"/CountHistos/");

    //grab filter count histograms  
    vector< string > filterCountMEs =  dbe_->getMEs();
    nTriggers_ = (int) filterCountMEs.size();    
    std::string tmpname = "";

    for(int trig = 0; trig < nTriggers_; trig++){
      tmpname = indir_ + "/CountHistos/" + filterCountMEs[trig];
      hSubFilterCount[trig]=dbe_->get(tmpname);
    }

    tmpname = indir_ + "/PassingBits_Summary";
    hCountSummary = dbe_->get(tmpname);

    //book efficiency histograms
    dbe_->setCurrentFolder(outdir_);
    TH1F* refhisto;   
    int nbin_sub;
    
    for(int trig = 0; trig < nTriggers_; trig++){
      if(hSubFilterCount[trig]){
	refhisto = hSubFilterCount[trig]->getTH1F();
	nbin_sub = refhisto->GetNbinsX();
	//cout << "nbin_sub = " << nbin_sub << endl;
	//cout << "client: " << filterCountMEs[trig] << endl;
	hSubFilterEfficiency[trig] = dbe_->book1D("Efficiency_"+filterCountMEs[trig], 
						  "Efficiency_"+filterCountMEs[trig], 
						nbin_sub, 0.5, 0.5+(double)nbin_sub);
	for(int i = 1; i <= nbin_sub; i++)
	  hSubFilterEfficiency[trig]->getTH1F()->GetXaxis()->SetBinLabel(i, refhisto->GetXaxis()->GetBinLabel(i));
      }
    }
    if(hCountSummary){
      refhisto = hCountSummary->getTH1F();
      nbin_sub = refhisto->GetNbinsX();
      hEffSummary = dbe_->book1D("Efficiency_PassingBits_Summary",
				 "Efficiency_PassingBits_Summary",
				 nbin_sub, 0.5, 0.5+(double)nbin_sub);
      for(int i = 1; i <= nbin_sub; i++)
	hEffSummary->getTH1F()->GetXaxis()->SetBinLabel(i, refhisto->GetXaxis()->GetBinLabel(i));
    }
    
  }

}

//--------------------------------------------------------
void HLTMonMuonClient::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
   // optionally reset histograms here
}

void HLTMonMuonClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){

}

//--------------------------------------------------------
void HLTMonMuonClient::analyze(const Event& e, const EventSetup& context){

  TH1F * refhisto;
  int nbin_sub;
  for( int n = 0; n < nTriggers_; n++) {
    if(hSubFilterCount[n]){
      refhisto = hSubFilterCount[n]->getTH1F();
      nbin_sub = refhisto->GetNbinsX();
      for( int i = 1; i <= nbin_sub; i++ ) {
	double denominator = refhisto->GetBinContent(0); 
	double numerator = refhisto->GetBinContent(i);
	//cout << refhisto->GetXaxis()->GetBinLabel(i) << " " << numerator << " " << denominator << endl;
	double eff = 0.0;
	if( denominator != 0 ) eff = numerator/denominator;
	hSubFilterEfficiency[n]->setBinContent(i, eff);
	//hSubFilterEfficiency[n]->Fill(i, eff);
      }
    }
  }
  
  if(hCountSummary ){                                                                                                                               
    refhisto = hCountSummary->getTH1F();
    nbin_sub = refhisto->GetNbinsX();
    for( int i = 0; i < nbin_sub; i++ ) {
      double denominator = refhisto->GetBinContent(1); // HLT_L1MuOpen usually. If not , the lowest threshold HLT_L1Mu*
      double numerator = refhisto->GetBinContent(i+1);
      double eff = 0.0;
      if( denominator != 0 ) eff = numerator/denominator;
      hEffSummary->setBinContent(i+1, eff);
    }
  }
  

/*------------- 
   counterEvt++;
   if (prescaleEvt<1) return;
   if (prescaleEvt>0 && counterEvt%prescaleEvt!=0) return;

   // The code below duplicates one from endLuminosityBlock function
   vector<string> meVec = dbe->getMEs();
   for(vector<string>::const_iterator it=meVec.begin(); it!=meVec.end(); it++){
     string full_path = input_dir + "/" + (*it);
     MonitorElement *me =dbe->get(full_path);
     if( !me ){
        LogError("TriggerDQM")<<full_path<<" NOT FOUND.";
        continue;
     }
     //  But for now we only do a simple workaround
     if( (*it) != "CSCTF_errors" ) continue;
     TH1F *errors = me->getTH1F();
     csctferrors_->getTH1F()->Reset();
     if(!errors) continue;
     for(int bin=1; bin<=errors->GetXaxis()->GetNbins(); bin++)
        csctferrors_->Fill(bin-0.5,errors->GetBinContent(bin));
   }
-----------------*/
}

//--------------------------------------------------------
void HLTMonMuonClient::endRun(const Run& r, const EventSetup& context){}

//--------------------------------------------------------
void HLTMonMuonClient::endJob(void){}

