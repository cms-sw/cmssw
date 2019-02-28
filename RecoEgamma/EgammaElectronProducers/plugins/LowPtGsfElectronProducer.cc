
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronProducer.h"

LowPtGsfElectronProducer::LowPtGsfElectronProducer( const edm::ParameterSet& cfg, 
						    const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc)
{}

LowPtGsfElectronProducer::~LowPtGsfElectronProducer()
{}

void LowPtGsfElectronProducer::produce( edm::Event& event, const edm::EventSetup& setup )
{
  reco::GsfElectronCollection electrons;
  algo_->completeElectrons(electrons, event, setup, globalCache());
  fillEvent(electrons, event);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronProducer);
