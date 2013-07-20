//
// $Id: DuplicatedElectronCleaner.cc,v 1.5 2013/02/27 23:26:56 wmtan Exp $
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
  \version  $Id: DuplicatedElectronCleaner.cc,v 1.5 2013/02/27 23:26:56 wmtan Exp $
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
//#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/PatUtils/interface/DuplicatedElectronRemover.h"

namespace pat{
  class DuplicatedElectronCleaner : public edm::EDProducer{
  public:
    explicit DuplicatedElectronCleaner(const edm::ParameterSet & iConfig);
    ~DuplicatedElectronCleaner();  
    
    virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;
    virtual void endJob();
    
  private:
    edm::InputTag electronSrc_;
    pat::DuplicatedElectronRemover duplicateRemover_; 
    uint64_t try_, pass_;
  };
} // namespace

pat::DuplicatedElectronCleaner::DuplicatedElectronCleaner(const edm::ParameterSet & iConfig):
  electronSrc_(iConfig.getParameter<edm::InputTag>("electronSource")),
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
pat::DuplicatedElectronCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  using namespace edm;
  Handle<View<reco::GsfElectron> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);
  try_ += electrons->size();
  
  //std::auto_ptr<RefVector<reco::GsfElectronCollection> > result(new RefVector<reco::GsfElectronCollection>());
  std::auto_ptr<RefToBaseVector<reco::GsfElectron> > result(new RefToBaseVector<reco::GsfElectron>());
  //std::auto_ptr<PtrVector<reco::GsfElectron> > result(new PtrVector<reco::GsfElectron>());
  std::auto_ptr< std::vector<size_t> > duplicates = duplicateRemover_.duplicatesToRemove(*electrons);
  
  std::vector<size_t>::const_iterator itdup = duplicates->begin(), enddup = duplicates->end();
  for (size_t i = 0, n = electrons->size(); i < n; ++i) {
    while ((itdup != enddup) && (*itdup < i)) { ++itdup; }
    if ((itdup != enddup) && (*itdup == i)) continue;
    //result->push_back(electrons->refAt(i).castTo<edm::Ref<reco::GsfElectronCollection> >());
    result->push_back(electrons->refAt(i));
    //result->push_back(electrons->ptrAt(i));
  }
  pass_ += result->size(); 
  iEvent.put(result);
}

void 
pat::DuplicatedElectronCleaner::endJob() 
{ 
}

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::DuplicatedElectronCleaner;
DEFINE_FWK_MODULE(DuplicatedElectronCleaner);
