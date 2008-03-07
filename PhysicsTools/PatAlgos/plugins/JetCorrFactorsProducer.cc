//
// $Id$
//

#include "PhysicsTools/PatAlgos/plugins/JetCorrFactorsProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/View.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/Jet.h"


#include <vector>
#include <memory>


using namespace pat;


JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& iConfig) {
  // initialize the configurables
  jetsSrc_ = iConfig.getParameter<edm::InputTag>( "jetSource" );

  // produces valuemap of jet correction factors
  produces<JetCorrFactorsMap>();
}


JetCorrFactorsProducer::~JetCorrFactorsProducer() {
}


void JetCorrFactorsProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // define the jet correctors - FIXME: make configurable
  const JetCorrector * defaultJetCorr = JetCorrector::getJetCorrector("MCJetCorrectorIcone5", iSetup);
  const JetCorrector * udsJetCorr     = JetCorrector::getJetCorrector("L5FlavorJetCorrectorUds", iSetup);
  const JetCorrector * gluJetCorr     = JetCorrector::getJetCorrector("L5FlavorJetCorrectorGluon", iSetup);
  const JetCorrector * cJetCorr       = JetCorrector::getJetCorrector("L5FlavorJetCorrectorC", iSetup);
  const JetCorrector * bJetCorr       = JetCorrector::getJetCorrector("L5FlavorJetCorrectorB", iSetup);

  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(jetsSrc_, jets);

  // loop over jets and retrieve the correction factors
  std::vector<JetCorrFactors> jetCorrs;
  for (edm::View<reco::Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {
    // retrieve the energy correction factors
    float scaleDefault = defaultJetCorr->correction(*itJet);
    // scale the jet; needed because subsequent corrections are to be applied after
    reco::Particle::LorentzVector aJet(scaleDefault*itJet->px(), scaleDefault*itJet->py(), scaleDefault*itJet->pz(), scaleDefault*itJet->energy());
    // create the actual object with scalefactos we want the valuemap to refer to
    JetCorrFactors aJetCorr(scaleDefault,
                            scaleDefault * udsJetCorr->correction(aJet),
                            scaleDefault * gluJetCorr->correction(aJet),
                            scaleDefault * cJetCorr->correction(aJet),
                            scaleDefault * bJetCorr->correction(aJet));
    jetCorrs.push_back(aJetCorr);
  }

  // build the value map
  std::auto_ptr<JetCorrFactorsMap> jetCorrsMap(new JetCorrFactorsMap());
  JetCorrFactorsMap::Filler filler(*jetCorrsMap);
  // jets and jetCorrs have their indices aligned by construction
  filler.insert(jets, jetCorrs.begin(), jetCorrs.end());
  filler.fill(); // do the actual filling

  // put our produced stuff in the event
  iEvent.put(jetCorrsMap);

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JetCorrFactorsProducer);
