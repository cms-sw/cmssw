/** \class CaloTowerCandidateCreator
 *
 * Framework module that produces a collection
 * of candidates with a CaloTowerCandidate compoment
 *
 * \author Luca Lista, INFN
 * modifyed by: F.Ratnikov UMd
 *
 *
 */
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <string>

class CaloTowerCandidateCreator : public edm::stream::EDProducer<> {
public:
  /// constructor from parameter set
  CaloTowerCandidateCreator(const edm::ParameterSet&);
  /// destructor
  ~CaloTowerCandidateCreator() override;

private:
  /// process one event
  void produce(edm::Event& e, const edm::EventSetup&) override;
  /// verbosity
  int mVerbose;
  /// token of source collection
  edm::EDGetTokenT<CaloTowerCollection> tok_src_;
  /// ET threshold
  double mEtThreshold;
  /// E threshold
  double mEThreshold;
};

#include "FWCore/Framework/interface/MakerMacros.h"
// remove following line after Jet/Met move to using
// exclusively CaloTowers
DEFINE_FWK_MODULE(CaloTowerCandidateCreator);

using namespace edm;
using namespace reco;
using namespace std;

CaloTowerCandidateCreator::CaloTowerCandidateCreator(const ParameterSet& p)
    : mVerbose(p.getUntrackedParameter<int>("verbose", 0)),
      mEtThreshold(p.getParameter<double>("minimumEt")),
      mEThreshold(p.getParameter<double>("minimumE")) {
  tok_src_ = consumes<CaloTowerCollection>(p.getParameter<edm::InputTag>("src"));

  produces<CandidateCollection>();
}

CaloTowerCandidateCreator::~CaloTowerCandidateCreator() {}

void CaloTowerCandidateCreator::produce(Event& evt, const EventSetup&) {
  Handle<CaloTowerCollection> caloTowers;
  evt.getByToken(tok_src_, caloTowers);

  auto cands = std::make_unique<CandidateCollection>();
  cands->reserve(caloTowers->size());
  unsigned idx = 0;
  for (; idx < caloTowers->size(); idx++) {
    const CaloTower* cal = &((*caloTowers)[idx]);
    if (mVerbose >= 2) {
      std::cout << "CaloTowerCandidateCreator::produce-> " << idx << " tower et/eta/phi/e: " << cal->et() << '/'
                << cal->eta() << '/' << cal->phi() << '/' << cal->energy() << " is...";
    }
    if (cal->et() >= mEtThreshold && cal->energy() >= mEThreshold) {
      math::PtEtaPhiMLorentzVector p(cal->et(), cal->eta(), cal->phi(), 0);
      RecoCaloTowerCandidate* c = new RecoCaloTowerCandidate(0, Candidate::LorentzVector(p));
      c->setCaloTower(CaloTowerRef(caloTowers, idx));
      cands->push_back(c);
      if (mVerbose >= 2)
        std::cout << "accepted: pT/eta/phi:" << c->pt() << '/' << c->eta() << '/' << c->phi() << std::endl;
    } else {
      if (mVerbose >= 2)
        std::cout << "rejected" << std::endl;
    }
  }
  if (mVerbose >= 1) {
    std::cout << "CaloTowerCandidateCreator::produce-> " << cands->size() << " candidates created" << std::endl;
  }
  evt.put(std::move(cands));
}
