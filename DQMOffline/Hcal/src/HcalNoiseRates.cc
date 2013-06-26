//
// HcalNoiseRates.cc
//
//   description: Calculation for single particle response corrections
//
//   author: K. Hatakeyama, H. Liu, Baylor
//
//

#include "DQMOffline/Hcal/interface/HcalNoiseRates.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// constructors and destructor
//

HcalNoiseRates::HcalNoiseRates(const edm::ParameterSet& iConfig)
{

  // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile","myfile.root");

  dbe_ = 0;
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
   
  Char_t histo[100];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("HcalNoiseRatesD/HcalNoiseRatesTask");
  }

  // set parameters
  rbxCollName_   = iConfig.getUntrackedParameter<edm::InputTag>("rbxCollName");
  minRBXEnergy_  = iConfig.getUntrackedParameter<double>("minRBXEnergy");
  minHitEnergy_  = iConfig.getUntrackedParameter<double>("minHitEnergy");

  useAllHistos_  = iConfig.getUntrackedParameter<bool>("useAllHistos", false);

  // book histograms

  //Lumi block is not drawn; the rest are
  if (useAllHistos_){
    sprintf  (histo, "hLumiBlockCount" );
    hLumiBlockCount_ = dbe_->book1D(histo, histo, 1, -0.5, 0.5);
  }
  
  sprintf  (histo, "hRBXEnergy" );
  hRBXEnergy_ = dbe_->book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType1" );
  hRBXEnergyType1_ = dbe_->book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType2" );
  hRBXEnergyType2_ = dbe_->book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType3" );
  hRBXEnergyType3_ = dbe_->book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXNHits" );
  hRBXNHits_ = dbe_->book1D(histo, histo, 73,-0.5,72.5);

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
  iEvent.getByLabel(rbxCollName_,handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HcalNoiseRBXCollection named " << rbxCollName_ << ".\n";
    return;
  }

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

  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseRates);
