//
// $Id: PATPhotonCleaner.h,v 1.2 2008/03/13 09:08:38 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATPhotonCleaner_h
#define PhysicsTools_PatAlgos_PATPhotonCleaner_h

/**
  \class    pat::PATPhotonCleaner PATPhotonCleaner.h "PhysicsTools/PatAlgos/interface/PATPhotonCleaner.h"
  \brief    Produces pat::Photon's

   The PATPhotonCleaner produces analysis-level pat::Photon's starting from
   a collection of objects of PhotonType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATPhotonCleaner.h,v 1.2 2008/03/13 09:08:38 llista Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
//#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"

#include "PhysicsTools/PatUtils/interface/DuplicatedPhotonRemover.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

namespace pat {

  template<typename PhotonIn, typename PhotonOut>
  class PATPhotonCleaner : public edm::EDProducer {
    public:
      enum RemovalAlgo { None, BySeed, BySuperCluster };
    
      explicit PATPhotonCleaner(const edm::ParameterSet & iConfig);
      ~PATPhotonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      // configurables
      edm::InputTag               photonSrc_;
      RemovalAlgo                 removeDuplicates_;
      RemovalAlgo                 removeElectrons_;
      std::vector<edm::InputTag>  electronsToCheck_;
        
      // helpers
      pat::helper::CleanerHelper<PhotonIn,
                                 PhotonOut,
                                 std::vector<PhotonOut>,
                                 GreaterByEt<PhotonOut> > helper_;
      pat::helper::MultiIsolator isolator_;

      // duplicate removal algo
      pat::DuplicatedPhotonRemover remover_;

      static RemovalAlgo fromString(const edm::ParameterSet & iConfig, const std::string &name);
      void removeDuplicates() ;
      void removeElectrons(const edm::Event &iEvent) ;
  };

  // now I'm typedeffing eveything, but I don't think we really need all them
  typedef PATPhotonCleaner<reco::Photon,reco::Photon>                   PATBasePhotonCleaner;
  //  typedef PATPhotonCleaner<reco::ConvertedPhoton,reco::ConvertedPhoton> PATConvertedPhotonCleaner;

}

template<typename PhotonIn, typename PhotonOut>
typename pat::PATPhotonCleaner<PhotonIn,PhotonOut>::RemovalAlgo
pat::PATPhotonCleaner<PhotonIn,PhotonOut>::fromString(const edm::ParameterSet & iConfig, 
        const std::string &parName) 
{
    std::string name = iConfig.getParameter<std::string>(parName);
    if (name == "none"  )         return None;
    if (name == "bySeed")         return BySeed;
    if (name == "bySuperCluster") return BySuperCluster;
    throw cms::Exception("Configuraton Error") << 
        "PATPhotonCleaner: " <<
        "Invalid choice '" << name <<"' for parameter " << name << ", valid options are " <<
        " 'none', 'bySeed', 'bySuperCluster'";
}

template<typename PhotonIn, typename PhotonOut>
pat::PATPhotonCleaner<PhotonIn,PhotonOut>::PATPhotonCleaner(const edm::ParameterSet & iConfig) :
  photonSrc_(iConfig.getParameter<edm::InputTag>( "photonSource" )),
  removeDuplicates_(fromString(iConfig, "removeDuplicates")),
  removeElectrons_( fromString(iConfig, "removeElectrons")),
  helper_(photonSrc_),
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet() )
{
  // produces vector of electrons
  produces<std::vector<PhotonOut> >();
  // producers also backmatch to the electrons
  produces<reco::CandRefValueMap>();

  if (removeElectrons_ != None) {
    if (!iConfig.exists("electrons")) throw cms::Exception("Configuraton Error") <<
        "PATPhotonCleaner: if using any electron removal, you have to specify" <<
        " the collection(s) of electrons, either as InputTag or VInputTag";
    std::vector<std::string> pars = iConfig.getParameterNamesForType<edm::InputTag>();
    if (std::find(pars.begin(), pars.end(), "electrons") != pars.end()) {
       electronsToCheck_.push_back(iConfig.getParameter<edm::InputTag>("electrons"));
    } else {
       electronsToCheck_ = iConfig.getParameter<std::vector<edm::InputTag> >("electrons");
    }
  }
}


template<typename PhotonIn, typename PhotonOut>
pat::PATPhotonCleaner<PhotonIn,PhotonOut>::~PATPhotonCleaner() {
}


template<typename PhotonIn, typename PhotonOut>
void pat::PATPhotonCleaner<PhotonIn,PhotonOut>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  // start a new event
  helper_.newEvent(iEvent);
  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  typedef typename edm::Ref< std::vector<PhotonIn> > PhotonInRef;
  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source photon
    const PhotonIn & srcPhoton = helper_.srcAt(idx);

    // clone the photon and convert it to the new type
    PhotonOut ourPhoton = static_cast<PhotonOut>(srcPhoton);

    // write the photon
    size_t selIdx = helper_.addItem(idx, ourPhoton);

    // test for isolation and set the bit if needed
    if (isolator_.enabled()) {
        uint32_t isolationWord = isolator_.test( helper_.source(), idx );
        helper_.addMark(selIdx, isolationWord);
    }

  }

  if (removeDuplicates_ != None) removeDuplicates();
  if (removeElectrons_  != None) removeElectrons(iEvent);

  helper_.done();
  if (isolator_.enabled()) isolator_.endEvent(); 
}

template<typename PhotonIn, typename PhotonOut>
void pat::PATPhotonCleaner<PhotonIn,PhotonOut>::removeElectrons(const edm::Event &iEvent) {
    uint32_t bit = 2; 
    typedef std::vector<edm::InputTag> VInputTag;
    for (VInputTag::const_iterator itt = electronsToCheck_.begin(), edt = electronsToCheck_.end();
                itt != edt; ++itt, bit <<= 1) {

        edm::Handle<edm::View<reco::RecoCandidate> > handle;
        iEvent.getByLabel(*itt, handle);

        std::auto_ptr< pat::OverlapList > electrons;
        if (removeElectrons_ == BySeed) {
            electrons = remover_.electronsBySeed(helper_.selected(), *handle);
        } else if (removeElectrons_ == BySuperCluster) {
            electrons = remover_.electronsBySuperCluster(helper_.selected(), *handle);
        }
        if (!electrons.get()) return;
        for (pat::OverlapList::const_iterator it = electrons->begin(),
                ed = electrons->end();
                it != ed;
                ++it) {
            size_t idx = it->first;
            helper_.setMark(idx, helper_.mark(idx) + bit);
        }
    }
}

template<typename PhotonIn, typename PhotonOut>
void pat::PATPhotonCleaner<PhotonIn,PhotonOut>::removeDuplicates() {
    std::auto_ptr< std::vector<size_t> > duplicates;
    if (removeDuplicates_ == BySeed) {
        duplicates = remover_.duplicatesBySeed(helper_.selected());
    } else if (removeDuplicates_ == BySuperCluster) {
        duplicates = remover_.duplicatesBySuperCluster(helper_.selected());
    }
    if (!duplicates.get()) return;
    for (std::vector<size_t>::const_iterator it = duplicates->begin(),
                                             ed = duplicates->end();
                                it != ed;
                                ++it) {
        helper_.setMark(*it, 1);
    }
}

#endif
