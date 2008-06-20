#include "PhysicsTools/PatAlgos/plugins/PATPhotonCleaner.h"

pat::PATPhotonCleaner::RemovalAlgo
pat::PATPhotonCleaner::fromString(const edm::ParameterSet & iConfig, 
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


pat::PATPhotonCleaner::PATPhotonCleaner(const edm::ParameterSet & iConfig) :
  photonSrc_(iConfig.getParameter<edm::InputTag>( "photonSource" )),
  removeDuplicates_(fromString(iConfig, "removeDuplicates")),
  removeElectrons_( fromString(iConfig, "removeElectrons")),
  helper_(photonSrc_),
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet() )
{
  helper_.configure(iConfig);      // learn whether to save good, bad, all, ...
  helper_.registerProducts(*this); // issue the produces<>() commands

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



pat::PATPhotonCleaner::~PATPhotonCleaner() {
}



void pat::PATPhotonCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  // start a new event
  helper_.newEvent(iEvent);
  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source photon
    const reco::Photon & srcPhoton = helper_.srcAt(idx);

    // clone the photon
    reco::Photon ourPhoton = srcPhoton;

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


void pat::PATPhotonCleaner::removeElectrons(const edm::Event &iEvent) {
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
            helper_.addMark(idx, bit);
        }
    }
}


void pat::PATPhotonCleaner::removeDuplicates() {
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
        helper_.addMark(*it, 1);
    }
}



void pat::PATPhotonCleaner::endJob() {
    edm::LogVerbatim("PATLayer0Summary|PATPhotonCleaner") << "PATPhotonCleaner end job. \n" <<
            "Input tag was " << photonSrc_.encode() <<
            "\nIsolation information:\n" <<
            isolator_.printSummary() <<
            "\nCleaner summary information:\n" <<
            helper_.printSummary();
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PATPhotonCleaner);
