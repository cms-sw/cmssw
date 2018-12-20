
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronProducer.h"

LowPtGsfElectronProducer::LowPtGsfElectronProducer( const edm::ParameterSet& cfg, 
						    const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc)
{}

LowPtGsfElectronProducer::~LowPtGsfElectronProducer()
{}

void LowPtGsfElectronProducer::beginEvent( edm::Event& event, 
					   const edm::EventSetup& setup )
{
  GsfElectronBaseProducer::beginEvent(event,setup);
}

void LowPtGsfElectronProducer::produce( edm::Event& event, const edm::EventSetup& setup )
{
  beginEvent(event,setup);
  algo_->completeElectrons(globalCache());
  fillEvent(event);
  endEvent();
}

//////////////////////////////////////////////////////////////////////////////////////////
//
//void LowPtGsfElectronProducer::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
//{
//  edm::ParameterSetDescription desc;
//  GsfElectronBaseProducer::fillDescription(desc); //@@ to be updated?
//  descriptions.add("lowPtGsfElectrons",desc);
//}
