//
// $Id: PATTauProducer.cc,v 1.9 2008/06/09 09:17:24 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"

#include <vector>
#include <memory>


using namespace pat;


PATTauProducer::PATTauProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  tauSrc_               = iConfig.getParameter<edm::InputTag>( "tauSource" );
  embedIsolationTracks_ = iConfig.getParameter<bool>         ( "embedIsolationTracks" );
  embedLeadTrack_       = iConfig.getParameter<bool>         ( "embedLeadTrack" );
  embedSignalTracks_    = iConfig.getParameter<bool>         ( "embedSignalTracks" );
  addGenMatch_    = iConfig.getParameter<bool>         ( "addGenMatch" );
  genMatchSrc_    = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  addTrigMatch_   = iConfig.getParameter<bool>               ( "addTrigMatch" );
  trigMatchSrc_   = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );
  addResolutions_ = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_      = iConfig.getParameter<bool>         ( "useNNResolutions" );
  tauResoFile_    = iConfig.getParameter<std::string>  ( "tauResoFile" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(tauResoFile_).fullPath(), useNNReso_);
  }

  // produces vector of taus
  produces<std::vector<Tau> >();
}


PATTauProducer::~PATTauProducer() {
  if (addResolutions_) delete theResoCalc_;
}


void PATTauProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  std::auto_ptr<std::vector<Tau> > patTaus(new std::vector<Tau>()); 

  edm::Handle<edm::View<TauType> > anyTaus;
  try {
    iEvent.getByLabel(tauSrc_, anyTaus);
  } catch (const edm::Exception &e) {
    edm::LogWarning("DataSource") << "WARNING! No Tau collection found. This missing input will not block the job. Instead, an empty tau collection is being be produced.";
    iEvent.put(patTaus);
    return;
  }
   
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genMatchSrc_, genMatch); 

  for (size_t idx = 0, ntaus = anyTaus->size(); idx < ntaus; ++idx) {
    edm::RefToBase<TauType> tausRef = anyTaus->refAt(idx);
    const TauType * originalTau = tausRef.get();

    Tau aTau(tausRef);
    if (embedIsolationTracks_) aTau.embedIsolationTracks();
    if (embedLeadTrack_) aTau.embedLeadTrack();
    if (embedSignalTracks_) aTau.embedSignalTracks();

    // store the match to the generated final state taus
    if (addGenMatch_) {
      reco::GenParticleRef genTau = (*genMatch)[tausRef];
      if (genTau.isNonnull() && genTau.isAvailable() ) {
        aTau.setGenLepton(*genTau);
      } // leave empty if no match found
    }
    
    // matches to trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigMatchSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[tausRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          aTau.addTriggerMatch(*trigPrim);
        }
      }
    }

    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(aTau);
    }

    patTaus->push_back(aTau);
  }

  // sort taus in pT
  std::sort(patTaus->begin(), patTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  iEvent.put(patTaus);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauProducer);


