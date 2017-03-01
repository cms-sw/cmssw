//
// HcalNoiseRates.cc
//
//   description: Calculation for single particle response corrections
//
//   author: K. Hatakeyama, H. Liu, Baylor
//
//

#include "DQMOffline/Hcal/interface/HcalNoiseRates.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// constructors and destructor
//

HcalNoiseRates::HcalNoiseRates(const edm::ParameterSet& iConfig)
{
   // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile","myfile.root");


  // set parameters
  rbxCollName_   = iConfig.getUntrackedParameter<edm::InputTag>("rbxCollName");
  tok_rbx_ = consumes<reco::HcalNoiseRBXCollection>(rbxCollName_);
  minRBXEnergy_  = iConfig.getUntrackedParameter<double>("minRBXEnergy");
  minHitEnergy_  = iConfig.getUntrackedParameter<double>("minHitEnergy");

  useAllHistos_  = iConfig.getUntrackedParameter<bool>("useAllHistos", false);

  //Hcal Noise Summary
  noisetoken_ = consumes<HcalNoiseSummary>(iConfig.getParameter<edm::InputTag>("noiselabel"));
}

  void HcalNoiseRates::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & /* iRun*/, edm::EventSetup const & /* iSetup */)

{ 
 
  
  ibooker.setCurrentFolder("HcalNoiseRatesD/HcalNoiseRatesTask");
  
  Char_t histo[100];

  // book histograms

  //Lumi block is not drawn; the rest are
  if (useAllHistos_){
    sprintf  (histo, "hLumiBlockCount" );
    hLumiBlockCount_ = ibooker.book1D(histo, histo, 1, -0.5, 0.5);
  }
  
  sprintf  (histo, "hRBXEnergy" );
  hRBXEnergy_ = ibooker.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType1" );
  hRBXEnergyType1_ = ibooker.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType2" );
  hRBXEnergyType2_ = ibooker.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType3" );
  hRBXEnergyType3_ = ibooker.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXNHits" );
  hRBXNHits_ = ibooker.book1D(histo, histo, 73,-0.5,72.5);

  //HcalNoiseSummary

  sprintf (histo, "nNNumChannels");
  nNNumChannels_ = ibooker.book1D(histo, histo, 100, 0, 100);
  sprintf (histo, "nNSumE");
  nNSumE_ = ibooker.book1D(histo, histo , 100, 0, 5000);
  sprintf (histo, "nNSumEt");
  nNSumEt_ = ibooker.book1D(histo, histo , 100, 0, 2000);

  sprintf (histo, "sNNumChannels");
  sNNumChannels_ = ibooker.book1D(histo, histo, 100, 0, 100);
  sprintf (histo, "sNSumE");
  sNSumE_ = ibooker.book1D(histo, histo , 100, 0, 5000);
  sprintf (histo, "sNSumEt");
  sNSumEt_ = ibooker.book1D(histo, histo , 100, 0, 2000);

  sprintf (histo, "iNNumChannels");
  iNNumChannels_ = ibooker.book1D(histo, histo, 100, 0, 100);
  sprintf (histo, "iNSumE");
  iNSumE_ = ibooker.book1D(histo, histo , 100, 0, 5000);
  sprintf (histo, "iNSumEt");
  iNSumEt_ = ibooker.book1D(histo, histo , 100, 0, 2000);

  sprintf (histo, "hNoise_maxZeros");
  hNoise_maxZeros_ = ibooker.book1D(histo, histo, 80, 0, 80);
  sprintf (histo, "hNoise_maxHPDHits");
  hNoise_maxHPDHits_ = ibooker.book1D(histo, histo, 20, 0, 20);
  sprintf (histo, "hNoise_maxHPDNoOtherHits");
  hNoise_maxHPDNoOtherHits_ = ibooker.book1D(histo, histo, 20, 0, 20);

}
  
  
HcalNoiseRates::~HcalNoiseRates()
{
}
  
  
//
// member functions
//
  
// ------------ method called to for each event  ------------
void
HcalNoiseRates::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

  // get the lumi section
  int lumiSection = iEvent.luminosityBlock();
  lumiCountMap_[lumiSection]++;

  // get the RBX Noise collection
  edm::Handle<reco::HcalNoiseRBXCollection> handle;
  iEvent.getByToken(tok_rbx_,handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HcalNoiseRBXCollection named " << rbxCollName_ << ".\n";
    return;
  }

  // get the Noise summary object
  edm::Handle<HcalNoiseSummary> summary_h;
  iEvent.getByToken(noisetoken_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    return;
  }
  const HcalNoiseSummary summary = *summary_h;

  //Fill the Noise Summary histograms
  nNNumChannels_->Fill(summary.numNegativeNoiseChannels());
  nNSumE_->Fill(summary.NegativeNoiseSumE());
  nNSumEt_->Fill(summary.NegativeNoiseSumEt());

  sNNumChannels_->Fill(summary.numSpikeNoiseChannels());
  sNSumE_->Fill(summary.spikeNoiseSumE());
  sNSumEt_->Fill(summary.spikeNoiseSumEt());

  iNNumChannels_->Fill(summary.numIsolatedNoiseChannels());
  iNSumE_->Fill(summary.isolatedNoiseSumE());
  iNSumEt_->Fill(summary.isolatedNoiseSumEt());

  hNoise_maxZeros_->Fill(summary.maxZeros());
  hNoise_maxHPDHits_->Fill(summary.maxHPDHits());
  hNoise_maxHPDNoOtherHits_->Fill(summary.maxHPDNoOtherHits());

  // loop over the RBXs and fill the histograms
  for(reco::HcalNoiseRBXCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const reco::HcalNoiseRBX &rbx=(*it);

    double energy = rbx.recHitEnergy(minHitEnergy_);

    int nhits = rbx.numRecHits(minHitEnergy_);

    if(energy < minRBXEnergy_) continue;

    hRBXEnergy_->Fill(energy);
    
    if      (nhits <= 9)  hRBXEnergyType1_->Fill(energy);
    else if (nhits <= 18) hRBXEnergyType2_->Fill(energy);
    else               	  hRBXEnergyType3_->Fill(energy);
    
    hRBXNHits_->Fill(nhits);
    
  }   // done looping over RBXs

}


// ------------ method called once each job just before starting event loop  ------------
void 
HcalNoiseRates::beginJob(){}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalNoiseRates::endJob() {

  if (useAllHistos_) hLumiBlockCount_->Fill(0.0, lumiCountMap_.size()); 

}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseRates);
