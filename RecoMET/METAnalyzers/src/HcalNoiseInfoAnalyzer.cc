//
// HcalNoiseInfoProducer.cc
//
//   description: Implementation of skeleton analyzer for the HCAL noise information
//
//   author: J.P. Chou, Brown
//
//

#include "RecoMET/METAnalyzers/interface/HcalNoiseInfoAnalyzer.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

using namespace reco;
  
//
// constructors and destructor
//
  
HcalNoiseInfoAnalyzer::HcalNoiseInfoAnalyzer(const edm::ParameterSet& iConfig)
{
  // set parameters
  rbxCollName_    = iConfig.getParameter<std::string>("rbxCollName");
  rootHistFilename_ = iConfig.getParameter<std::string>("rootHistFilename");
  noisetype_ = iConfig.getParameter<int>("noisetype");
}
  
  
HcalNoiseInfoAnalyzer::~HcalNoiseInfoAnalyzer()
{
}
  
  
//
// member functions
//
  
// ------------ method called to for each event  ------------
void
HcalNoiseInfoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
    
  Handle<HcalNoiseRBXCollection> handle;
  iEvent.getByLabel(rbxCollName_,handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HcalNoiseRBXCollection named " << rbxCollName_ << ".\n";
    return;
  }
    
  // loop over the RBXs
  for(HcalNoiseRBXCollection::const_iterator rit=handle->begin(); rit!=handle->end(); ++rit) {
    // get the rbx
    HcalNoiseRBX rbx=(*rit);
    HcalNoiseHPD maxhpd=(*rbx.maxHPD());

    // get the number of hits in the RBX to categorize the noise
    int numhits = rbx.numRecHits(1.5);
    if(numhits==0) continue;
    if(numhits>=19 && noisetype_!=2 && noisetype_!=3) continue;
    if(numhits>=9 && numhits<=18 && noisetype_!=1 && noisetype_!=3) continue;
    if(numhits<=8 && noisetype_!=0 && noisetype_!=3) continue;

    int failures=0;

    hMaxZeros_->Fill(rbx.maxZeros());
    hTotalZeros_->Fill(rbx.totalZeros());

    if(rbx.maxZeros()>3)   failures |= 0x1;
    if(rbx.totalZeros()>7) failures |= 0x2;

    double totale2ts=rbx.allChargeHighest2TS();
    double totale10ts=rbx.allChargeTotal();

    // loop over the HPDs in the RBX
    for(std::vector<HcalNoiseHPD>::const_iterator hit=rbx.HPDs().begin(); hit!=rbx.HPDs().end(); ++hit) {
      HcalNoiseHPD hpd=(*hit);

      // make sure we have at least 1 hit above 5 GeV
      if(hpd.numRecHits(5.0)<1) continue;

      double e2ts=hpd.bigChargeHighest2TS();
      double e10ts=hpd.bigChargeTotal();

      hE2ts_->Fill(e2ts);
      hE10ts_->Fill(e10ts);
      hE2tsOverE10ts_->Fill(e10ts ? e2ts/e10ts : -999);

      if(e10ts && !(e2ts/e10ts<0.95 && e2ts/e10ts>0.70)) failures |= 0x4;
      
      int numhits = hpd.numRecHits(1.5);

      hHPDNHits_->Fill(numhits);
      if(numhits>16) failures |= 0x8;
      
    }
    hRBXE2ts_->Fill(totale2ts);
    hRBXE10ts_->Fill(totale10ts);
    hRBXE2tsOverE10ts_->Fill(totale10ts ? totale2ts/totale10ts : -999);

    if(totale10ts && totale2ts/totale10ts<0.70) failures |= 0x10;

    hFailures_->Fill(failures);
    double energy=rbx.recHitEnergy(1.5);

    if(failures==0) hAfterRBXEnergy_->Fill(energy);
    hBeforeRBXEnergy_->Fill(energy);
  }

  return;
}


// ------------ method called once each job just before starting event loop  ------------
void 
HcalNoiseInfoAnalyzer::beginJob(const edm::EventSetup&)
{
  // book histograms
  rootfile_ = new TFile(rootHistFilename_.c_str(), "RECREATE");

  hMaxZeros_ = new TH1D("hMaxZeros","Max # of zeros in an RBX",15,-0.5,14.5);
  hTotalZeros_ = new TH1D("hTotalZeros","total # of zeros in an RBX",15,-0.5,14.5);
  hE2ts_ = new TH1D("hE2ts","E(2ts) for the highest energy digi in an HPD",100,0,10000);
  hE10ts_ = new TH1D("hE10ts","E(10ts) for the highest energy digi in an HPD",100,0,10000);
  hE2tsOverE10ts_ = new TH1D("hE2tsOverE10ts","E(t2s)/E(10ts) for the highest energy digi in an HPD",100,-5,5);
  hRBXE2ts_ = new TH1D("hRBXE2ts","Sum RBX E(2ts)",100,0,10000);
  hRBXE10ts_ = new TH1D("hRBXE10ts","Sum RBX E(10ts)",100,0,10000);
  hRBXE2tsOverE10ts_ = new TH1D("hRBXE2tsOverE10ts","Sum RBX E(t2s)/E(10ts)",100,-5,5);
  hHPDNHits_ = new TH1D("hHPDNHits","Number of Hits with E>1.5 GeV in an HPD",19,-0.5,18.5);
  
  hFailures_ = new TH1D("hFailures","code designating which cut the event failed (if any)",32,-0.5,31.5);
  hBeforeRBXEnergy_ = new TH1D("hBeforeRBXEnergy","Total RecHit Energy in RBX before cuts",100,0,1000);
  hAfterRBXEnergy_ = new TH1D("hAfterRBXEnergy","Total RecHit Energy in RBX after cuts",100,0,1000);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalNoiseInfoAnalyzer::endJob() {

  // write histograms
  rootfile_->cd();

  hMaxZeros_->Write();
  hTotalZeros_->Write();
  hE2ts_->Write();
  hE10ts_->Write();
  hE2tsOverE10ts_->Write();
  hRBXE2ts_->Write();
  hRBXE10ts_->Write();
  hRBXE2tsOverE10ts_->Write();
  hHPDNHits_->Write();
  
  hFailures_->Write();
  hBeforeRBXEnergy_->Write();
  hAfterRBXEnergy_->Write();

  rootfile_->Close();
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseInfoAnalyzer);
