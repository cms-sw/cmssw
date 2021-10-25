/** \class ShallowCloneProducer
 *
 * Clones a concrete Candidate collection
 * to a CandidateCollection (i.e.: OwnVector<Candidate>) filled
 * with shallow clones of the original candidate collection
 *
 * \author: Francesco Fabozzi, INFN
 *          modified by Luca Lista, INFN
 *
 * Template parameters:
 * - C : Concrete candidate collection type
 *
 */

#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

template <typename C>
class ShallowCloneProducer : public edm::global::EDProducer<> {
public:
  /// constructor from parameter set
  explicit ShallowCloneProducer(const edm::ParameterSet&);
  /// destructor
  ~ShallowCloneProducer() override;

private:
  /// process an event
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  /// labels of the collection to be converted
  const edm::EDGetTokenT<C> srcToken_;
};

template <typename C>
ShallowCloneProducer<C>::ShallowCloneProducer(const edm::ParameterSet& par)
    : srcToken_(consumes<C>(par.template getParameter<edm::InputTag>("src"))) {
  produces<reco::CandidateCollection>();
}

template <typename C>
ShallowCloneProducer<C>::~ShallowCloneProducer() {}

template <typename C>
void ShallowCloneProducer<C>::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const {
  std::unique_ptr<reco::CandidateCollection> coll(new reco::CandidateCollection);
  edm::Handle<C> masterCollection;
  evt.getByToken(srcToken_, masterCollection);
  for (size_t i = 0; i < masterCollection->size(); ++i) {
    reco::CandidateBaseRef masterClone(edm::Ref<C>(masterCollection, i));
    coll->push_back(new reco::ShallowCloneCandidate(masterClone));
  }
  evt.put(std::move(coll));
}

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

typedef ShallowCloneProducer<reco::GenMETCollection> GenMETShallowCloneProducer;
typedef ShallowCloneProducer<reco::GsfElectronCollection> PixelMatchGsfElectronShallowCloneProducer;
typedef ShallowCloneProducer<reco::MuonCollection> MuonShallowCloneProducer;
typedef ShallowCloneProducer<reco::CaloMETCollection> CaloMETShallowCloneProducer;
typedef ShallowCloneProducer<reco::ElectronCollection> ElectronShallowCloneProducer;
typedef ShallowCloneProducer<reco::GenJetCollection> GenJetShallowCloneProducer;
typedef ShallowCloneProducer<reco::CaloJetCollection> CaloJetShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenMETShallowCloneProducer);
DEFINE_FWK_MODULE(PixelMatchGsfElectronShallowCloneProducer);
DEFINE_FWK_MODULE(MuonShallowCloneProducer);
DEFINE_FWK_MODULE(CaloMETShallowCloneProducer);
DEFINE_FWK_MODULE(ElectronShallowCloneProducer);
DEFINE_FWK_MODULE(GenJetShallowCloneProducer);
DEFINE_FWK_MODULE(CaloJetShallowCloneProducer);
