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
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

class PFMET : public edm::global::EDProducer<> {
public:
  explicit PFMET(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  /// Input PFCandidates
  edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidates_;
  edm::EDPutTokenT<reco::METCollection> putToken_;

  pf2pat::PFMETAlgo pfMETAlgo_;
};

using namespace std;
using namespace edm;
using namespace reco;
using namespace math;

PFMET::PFMET(const edm::ParameterSet& iConfig) : pfMETAlgo_(iConfig) {
  auto inputTagPFCandidates = iConfig.getParameter<InputTag>("PFCandidates");
  tokenPFCandidates_ = consumes<PFCandidateCollection>(inputTagPFCandidates);

  putToken_ = produces<METCollection>();

  LogDebug("PFMET") << " input collection : " << inputTagPFCandidates;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMET);

void PFMET::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const {
  LogDebug("PFMET") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;

  // get PFCandidates
  METCollection output{1, pfMETAlgo_.produce(iEvent.get(tokenPFCandidates_))};
  iEvent.emplace(putToken_, std::move(output));

  LogDebug("PFMET") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;
}
