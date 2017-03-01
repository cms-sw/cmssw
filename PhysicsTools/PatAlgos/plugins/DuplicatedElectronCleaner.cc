//
//

/**
  \class    pat::DuplicatedElectronCleaner DuplicatedElectronCleaner.h "PhysicsTools/PatAlgos/interface/DuplicatedElectronCleaner.h"
  \brief    Remove duplicates from the list of electrons

   The DuplicatedElectronCleaner removes duplicates from the input collection.
   Two electrons are considered duplicate if they share the same gsfTrack or the same superCluster.
   Among the two, the one with |E/P| nearest to 1 is kept.
   This is performed by the DuplicatedElectronRemover in PhysicsTools/PatUtils

   The output is an edm::RefVector<reco:::GsfElectron>,
   which can be read through edm::View<reco::GsfElectron>

  \author   Giovanni Petrucciani
  \version  $Id: DuplicatedElectronCleaner.cc,v 1.4 2010/02/20 21:00:16 wmtan Exp $
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
//#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

namespace pat{
  class DuplicatedElectronCleaner : public edm::global::EDProducer<> {
  public:
    explicit DuplicatedElectronCleaner(const edm::ParameterSet & iConfig);
    ~DuplicatedElectronCleaner();

    virtual void produce(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override final;

  private:
    const edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronSrcToken_;
    const pat::DuplicatedElectronRemover duplicateRemover_;
    mutable std::atomic<uint64_t> try_, pass_;
  };
} // namespace

pat::DuplicatedElectronCleaner::DuplicatedElectronCleaner(const edm::ParameterSet & iConfig):
  electronSrcToken_(consumes<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>("electronSource"))),
  duplicateRemover_(),
  try_(0), pass_(0)
{
  //produces<edm::RefVector<reco::GsfElectronCollection> >();
  produces<edm::RefToBaseVector<reco::GsfElectron> >();
  //produces<edm::PtrVector<reco::GsfElectron> >();
}

pat::DuplicatedElectronCleaner::~DuplicatedElectronCleaner()
{
}

void
pat::DuplicatedElectronCleaner::produce(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const 
{
  using namespace edm;
  Handle<View<reco::GsfElectron> > electrons;
  iEvent.getByToken(electronSrcToken_, electrons);
  try_ += electrons->size();

  //auto result = std::make_unique<RefVector<reco::GsfElectronCollection>>();
  auto result = std::make_unique<RefToBaseVector<reco::GsfElectron>>();
  //auto result = std::make_unique<PtrVector<reco::GsfElectron>>();
  std::unique_ptr< std::vector<size_t> > duplicates = duplicateRemover_.duplicatesToRemove(*electrons);

  std::vector<size_t>::const_iterator itdup = duplicates->begin(), enddup = duplicates->end();
  for (size_t i = 0, n = electrons->size(); i < n; ++i) {
    while ((itdup != enddup) && (*itdup < i)) { ++itdup; }
    if ((itdup != enddup) && (*itdup == i)) continue;
    //result->push_back(electrons->refAt(i).castTo<edm::Ref<reco::GsfElectronCollection> >());
    result->push_back(electrons->refAt(i));
    //result->push_back(electrons->ptrAt(i));
  }
  pass_ += result->size();
  iEvent.put(std::move(result));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::DuplicatedElectronCleaner;
DEFINE_FWK_MODULE(DuplicatedElectronCleaner);
