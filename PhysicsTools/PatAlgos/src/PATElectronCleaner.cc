//
// $Id: PATElectronCleaner.cc,v 1.4 2008/01/24 09:20:58 fronga Exp $
//

#include "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"


#include <vector>
#include <memory>


using pat::PATElectronCleaner;


PATElectronCleaner::PATElectronCleaner(const edm::ParameterSet & iConfig) :
    electronSrc_(iConfig.getParameter<edm::InputTag>("electronSource")),
    removeDuplicates_(iConfig.getParameter<bool>("removeDuplicates")),
    helper_(electronSrc_),
    selectionCfg_(iConfig.getParameter<edm::ParameterSet>("selection")),
    selectionType_(selectionCfg_.getParameter<std::string>("type"))
{
  // produces vector of electrons
  produces<std::vector<reco::PixelMatchGsfElectron> >();

  // produces also backmatch to the original electrons
  produces<reco::CandRefValueMap>();

  // Create electron selector if requested
  doSelection_ = ( selectionType_ != "none" );
  if ( doSelection_ ) {
    selector_ = std::auto_ptr<ElectronSelector>( new ElectronSelector(selectionCfg_) );
  }

}


PATElectronCleaner::~PATElectronCleaner() {
}


void PATElectronCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // start a new event
  helper_.newEvent(iEvent);

  // Get electron IDs if needed
  edm::Handle<reco::ElectronIDAssociationCollection> electronIDs;
  if ( doSelection_ && selectionType_ != "custom" ) {
    iEvent.getByLabel( selectionCfg_.getParameter<edm::InputTag>("eIDsource"), 
                       electronIDs );
  }

  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source electron
    const reco::PixelMatchGsfElectron &srcElectron = helper_.srcAt(idx);    

    // perform the selection
    if ( doSelection_ &&
         !selector_->filter(idx,helper_.source(),(*electronIDs)) ) continue;

    // clone the electron so we can modify it (if we want)
    reco::PixelMatchGsfElectron ourElectron = srcElectron; 

    // write the muon
    helper_.addItem(idx, ourElectron); 
  }

  // remove ghosts, by marking them
  if (removeDuplicates_) { 
    removeDuplicates(); 
  }

  // tell him that we're done.
  helper_.done(); // he does event.put by itself
}

/* --- Original comment from TQAF follows ----
 * it is possible that there are multiple electron objects in the collection that correspond to the same
 * real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
 *  (i would guess the latter doesn't actually happen).
 * NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly   
 */
void PATElectronCleaner::removeDuplicates() {
    std::auto_ptr< std::vector<size_t> > duplicates = 
      duplicateRemover_.duplicatesToRemove(helper_.selected());
    for (std::vector<size_t>::const_iterator it = duplicates->begin(),
                                             ed = duplicates->end();
                                it != ed;
                                ++it) {
        helper_.setMark(*it, 1);
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATElectronCleaner);
