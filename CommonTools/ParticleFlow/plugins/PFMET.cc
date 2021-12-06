/**\class PFMET
\brief Computes the MET from a collection of PFCandidates. HF missing!

\todo Add HF energy to the MET calculation (access HF towers)

\author Colin Bernet
\date   february 2008
*/

#include "CommonTools/ParticleFlow/interface/PFMETAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

class PFMET : public edm::EDProducer {
public:
  explicit PFMET(const edm::ParameterSet&);

  ~PFMET() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginJob() override;

private:
  /// Input PFCandidates
  edm::InputTag inputTagPFCandidates_;
  edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidates_;

  pf2pat::PFMETAlgo pfMETAlgo_;
};

using namespace std;
using namespace edm;
using namespace reco;
using namespace math;

PFMET::PFMET(const edm::ParameterSet& iConfig) : pfMETAlgo_(iConfig) {
  inputTagPFCandidates_ = iConfig.getParameter<InputTag>("PFCandidates");
  tokenPFCandidates_ = consumes<PFCandidateCollection>(inputTagPFCandidates_);

  produces<METCollection>();

  LogDebug("PFMET") << " input collection : " << inputTagPFCandidates_;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMET);

PFMET::~PFMET() {}

void PFMET::beginJob() {}

void PFMET::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFMET") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;

  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_, pfCandidates);

  unique_ptr<METCollection> pOutput(new METCollection());

  pOutput->push_back(pfMETAlgo_.produce(*pfCandidates));
  iEvent.put(std::move(pOutput));

  LogDebug("PFMET") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;
}
