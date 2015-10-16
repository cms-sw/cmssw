
#include "GEDGsfElectronProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"

#include <iostream>
#include <string>

using namespace reco;

GEDGsfElectronProducer::GEDGsfElectronProducer( const edm::ParameterSet & cfg, const gsfAlgoHelpers::HeavyObjectCache* hoc )
  : GsfElectronBaseProducer(cfg,hoc)
 {
   egmPFCandidateCollection_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("egmPFCandidatesTag"));
   outputValueMapLabel_ = cfg.getParameter<std::string>("outputEGMPFValueMap");

   produces<edm::ValueMap<reco::GsfElectronRef> >(outputValueMapLabel_);
}

GEDGsfElectronProducer::~GEDGsfElectronProducer()
 {}

// ------------ method called to produce the data  ------------
void GEDGsfElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {
  beginEvent(event,setup) ;
  matchWithPFCandidates(event);
  algo_->completeElectrons(globalCache()) ;
  algo_->setMVAOutputs(globalCache(),gsfMVAOutputMap_);
  algo_->setMVAInputs(gsfMVAInputMap_);
  fillEvent(event) ;

  // ValueMap
  std::auto_ptr<edm::ValueMap<reco::GsfElectronRef> > valMap_p(new edm::ValueMap<reco::GsfElectronRef>);
  edm::ValueMap<reco::GsfElectronRef>::Filler valMapFiller(*valMap_p);
  fillGsfElectronValueMap(event,valMapFiller);
  valMapFiller.fill();
  event.put(valMap_p,outputValueMapLabel_);  
  // Done with the ValueMap

  endEvent() ;
 }

void GEDGsfElectronProducer::fillGsfElectronValueMap(edm::Event & event, edm::ValueMap<reco::GsfElectronRef>::Filler & filler)
{
  // Read the collection of PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  
  bool found = event.getByToken(egmPFCandidateCollection_, pfCandidates);
  if(!found) {
    edm::LogError("GEDGsfElectronProducer")
       <<" cannot get PFCandidates! ";
  }

  //Loop over the collection of PFFCandidates
  reco::PFCandidateCollection::const_iterator it = pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator itend = pfCandidates->end() ;
  std::vector<reco::GsfElectronRef> values;

  for ( ; it != itend ; ++it) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if( it->gsfTrackRef().isNonnull()) {
      // now look for the corresponding GsfElectron
      GsfElectronEqual myEqual(it->gsfTrackRef());
      const reco::GsfElectronCollection::const_iterator itcheck=
	std::find_if(orphanHandle()->begin(),orphanHandle()->end(),myEqual);
      if (itcheck != orphanHandle()->end()) {
	// Build the Ref from the handle and the index
	myRef = reco::GsfElectronRef(orphanHandle(),itcheck-orphanHandle()->begin());
      }
    }
    values.push_back(myRef);
  }
  filler.insert(pfCandidates,values.begin(),values.end());
}


// Something more clever has to be found. The collections are small, so the timing is not 
// an issue here; but it is clearly suboptimal

void GEDGsfElectronProducer::matchWithPFCandidates(edm::Event & event)
{
  gsfMVAInputMap_.clear();
  gsfMVAOutputMap_.clear();

  // Read the collection of PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  
  bool found = event.getByToken(egmPFCandidateCollection_, pfCandidates);
  if(!found) {
    edm::LogError("GEDGsfElectronProducer")
       <<" cannot get PFCandidates! ";
  }

  //Loop over the collection of PFFCandidates
  reco::PFCandidateCollection::const_iterator it = pfCandidates->begin();
  reco::PFCandidateCollection::const_iterator itend = pfCandidates->end() ;
  
  for ( ; it != itend ; ++it) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if( it->gsfTrackRef().isNonnull()) {

      reco::GsfElectron::MvaOutput myMvaOutput;
      // at the moment, undefined
      myMvaOutput.status = it->egammaExtraRef()->electronStatus() ;
      gsfMVAOutputMap_[it->gsfTrackRef()] = myMvaOutput;

      reco::GsfElectron::MvaInput myMvaInput;
      myMvaInput.earlyBrem = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_FirstBrem);
      myMvaInput.lateBrem = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_LateBrem);
      myMvaInput.deltaEta = it->egammaExtraRef()->mvaVariable(reco::PFCandidateEGammaExtra::MVA_DeltaEtaTrackCluster);
      myMvaInput.sigmaEtaEta = it->egammaExtraRef()->sigmaEtaEta();
      myMvaInput.hadEnergy = it->egammaExtraRef()->hadEnergy();
      gsfMVAInputMap_[it->gsfTrackRef()] = myMvaInput;
    }
  }
}
