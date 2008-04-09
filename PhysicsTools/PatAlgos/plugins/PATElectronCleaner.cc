//
// $Id: PATElectronCleaner.cc,v 1.3 2008/03/14 15:13:50 gpetrucc Exp $
//
#include "PhysicsTools/PatAlgos/plugins/PATElectronCleaner.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include <vector>
#include <memory>
#include <sstream>

using pat::PATElectronCleaner;

PATElectronCleaner::PATElectronCleaner(const edm::ParameterSet & iConfig) :
    electronSrc_(iConfig.getParameter<edm::InputTag>("electronSource")),
    removeDuplicates_(iConfig.getParameter<bool>("removeDuplicates")),
    helper_(electronSrc_),
    isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet() ),
    selectionCfg_(iConfig.getParameter<edm::ParameterSet>("selection")),
    selectionType_(selectionCfg_.getParameter<std::string>("type")),
    selector_(reco::modules::make<ElectronSelector>(selectionCfg_)) 
{
  helper_.configure(iConfig);      // learn whether to save good, bad, all, ...
  helper_.registerProducts(*this); // issue the produces<>() commands
}

PATElectronCleaner::~PATElectronCleaner() {
}

void PATElectronCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // start a new event
  helper_.newEvent(iEvent);

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  // Get additional info from the event, if needed
  const reco::ClusterShape* clusterShape = 0;
  edm::Handle<reco::ElectronIDAssociationCollection> electronIDs;
  if ( selectionType_ != "none" && selectionType_ != "custom" ) {
    iEvent.getByLabel( selectionCfg_.getParameter<edm::InputTag>("eIdSource"), 
                       electronIDs );
  } 

  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {

    // read the source electron
    const reco::GsfElectron &srcElectron = helper_.srcAt(idx);    

    // clone the electron so we can modify it (if we want)
    reco::GsfElectron ourElectron = srcElectron; 

    // Add the electron to the working collection
    size_t selIdx = helper_.addItem(idx, ourElectron);

    // get the cluster shape for this electron selection
    if ( selectionType_ == "custom" ) {
      const reco::ClusterShapeRef& shapeRef = getClusterShape_( &srcElectron, iEvent);
      clusterShape = &(*shapeRef);
    }

    // apply selection and set bits accordingly
    if ( selector_.filter(idx,helper_.source(),(*electronIDs),clusterShape) ) {
        helper_.addMark(selIdx, pat::Flags::Selection::Bit0); // opaque, at the moment
    }

    // test for isolation and set the bit if needed
    if (isolator_.enabled()) {
        uint32_t isolationWord = isolator_.test( helper_.source(), idx );
        helper_.addMark(selIdx, isolationWord);
    }

  }

  // remove ghosts, by marking them
  if (removeDuplicates_) { 
    removeDuplicates(); 
  }

  // tell him that we're done. 
  helper_.done(); // he does event.put by itself
  if (isolator_.enabled()) isolator_.endEvent();
}


/* --- Original comment from TQAF follows ----
 * it is possible that there are multiple electron objects in the collection that correspond to the same
 * real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
 *  (i would guess the latter doesn't actually happen).
 * NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly   
 */
void PATElectronCleaner::removeDuplicates() {
    // we must use 'accepted()', as we don't want to kill a good electron because it overlaps with the bad one.
    MyCleanerHelper::FilteredCollection accepted = helper_.accepted(); 
    std::auto_ptr< std::vector<size_t> > duplicates = duplicateRemover_.duplicatesToRemove(accepted);
    for (std::vector<size_t>::const_iterator it = duplicates->begin(),
                                             ed = duplicates->end();
                                it != ed;
                                ++it) {
        helper_.addMark(accepted.originalIndexOf(*it), pat::Flags::Core::Duplicate);
    }
}

// Only needed until clustershape is inside Electron (should come in 2_0_0)
const reco::ClusterShapeRef& 
PATElectronCleaner::getClusterShape_( const reco::GsfElectron* electron, 
                                      const edm::Event&        event
                                    ) const {
  // Get association maps linking BasicClusters to ClusterShape.
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleBarrel;
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleEndcap;
  event.getByLabel( selectionCfg_.getParameter<edm::InputTag>("clusterShapeBarrel"),
                    clusterShapeHandleBarrel );
  event.getByLabel( selectionCfg_.getParameter<edm::InputTag>("clusterShapeEndcap"),
                    clusterShapeHandleEndcap );

  // Find entry in map corresponding to seed BasicCluster of SuperCluster
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
  if (electron->classification()<100) 
    {
      seedShpItr=clusterShapeHandleBarrel->find(electron->superCluster()->seed());
      if (electron->classification()==40 && seedShpItr == clusterShapeHandleBarrel->end()) 
        seedShpItr = clusterShapeHandleEndcap->find(electron->superCluster()->seed());
    } 
  else 
    {
      seedShpItr = clusterShapeHandleEndcap->find(electron->superCluster()->seed());
    }

  return seedShpItr->val;
}

void PATElectronCleaner::endJob() { 
    edm::LogVerbatim("PATLayer0Summary|PATElectronCleaner") << "PATElectronCleaner end job. \n" <<
            "Input tag was " << electronSrc_.encode() <<
            "\nIsolation information:\n" <<
            isolator_.printSummary() <<
            "\nCleaner summary information:\n" <<
            helper_.printSummary();
    helper_.endJob(); 
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronCleaner);
