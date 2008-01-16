//
// $Id: PATElectronCleaner.cc,v 1.1 2008/01/16 01:20:42 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/interface/PATElectronCleaner.h"

#include <vector>
#include <memory>


using pat::PATElectronCleaner;


PATElectronCleaner::PATElectronCleaner(const edm::ParameterSet & iConfig) :
    electronSrc_(iConfig.getParameter<edm::InputTag>("electronSource")),
    removeDuplicates_(iConfig.getParameter<bool>("removeDuplicates")),
    helper_(electronSrc_) 
{
  // produces vector of electrons
  produces<std::vector<reco::PixelMatchGsfElectron> >();

  // producers also backmatch to the electrons
  produces<reco::CandRefValueMap>();
}


PATElectronCleaner::~PATElectronCleaner() {
}


void PATElectronCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // start a new event
  helper_.newEvent(iEvent);
  
  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source electron
    const reco::PixelMatchGsfElectron &srcElectron = helper_.srcAt(idx);    

    // clone the electron so we can modify it (if we want)
    reco::PixelMatchGsfElectron ourElectron = srcElectron; 

    // perform the selection
    if (false) continue; // now there is no real selection for electrons (except for duplicate removal below)

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

/* --- Comment by Giovanni Petrucciani, when porting to PAT Cleaners ---
 * Now it directly manipulater the "helper_" to mark items not to be saved 
 */
/* --- Original comment from TQAF follows ----
 * it is possible that there are multiple electron objects in the collection that correspond to the same
 * real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
 *  (i would guess the latter doesn't actually happen).
 * NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly   
 */
void PATElectronCleaner::removeDuplicates() {
  size_t size = helper_.size();
  for (size_t ie = 0; ie < size; ++ie) {
      if (helper_.mark(ie) != 0) continue; // if already marked bad

      reco::GsfTrackRef thistrack  = helper_[ie].gsfTrack();
      reco::SuperClusterRef thissc = helper_[ie].superCluster();

      for (size_t je = ie+1; je < size; ++je) {
          if (helper_.mark(je) != 0) continue; // if already marked bad

          if ( ( thistrack == helper_[je].gsfTrack()) ||
                  (thissc  == helper_[je].superCluster()) ) {
              // we have a match, arbitrate and mark one for removal
              // keep the one with E/P closer to unity
              float diff1 = fabs(helper_[ie].eSuperClusterOverP()-1);
              float diff2 = fabs(helper_[je].eSuperClusterOverP()-1);

              if (diff1<diff2) {
                  helper_.setMark(je, 1);
              } else {
                  helper_.setMark(ie, 1);
              }
          }
      }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(pat::PATElectronCleaner);
