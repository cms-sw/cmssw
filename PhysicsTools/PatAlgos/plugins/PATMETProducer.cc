//
// $Id$
//

#include "PhysicsTools/PatAlgos/plugins/PATMETProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"

#include <memory>


using namespace pat;


PATMETProducer::PATMETProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  metSrc_         = iConfig.getParameter<edm::InputTag>("metSource");
  addGenMET_      = iConfig.getParameter<bool>         ("addGenMET");
  genMETSrc_      = iConfig.getParameter<edm::InputTag>("genMETSource");
  addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");
  useNNReso_      = iConfig.getParameter<bool>         ("useNNResolutions");
  metResoFile_    = iConfig.getParameter<std::string>  ("metResoFile");
  
  // construct resolution calculator
  if (addResolutions_) metResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(metResoFile_).fullPath(), useNNReso_);
  
  // produces vector of mets
  produces<std::vector<MET> >();
}


PATMETProducer::~PATMETProducer() {
  if (addResolutions_) delete metResoCalc_;
}


void PATMETProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of MET's from the event
  edm::Handle<edm::View<METType> > mets;
  iEvent.getByLabel(metSrc_, mets);

  // Get the vector of generated met from the event if needed
  edm::Handle<edm::View<reco::GenMET> > genMETs;
  if (addGenMET_) {
    iEvent.getByLabel(genMETSrc_, genMETs);
  }

  // loop over mets
  std::vector<MET> * patMETs = new std::vector<MET>(); 
  for (edm::View<METType>::const_iterator itMET = mets->begin(); itMET != mets->end(); itMET++) {
    // construct the MET from the ref -> save ref to original object
    unsigned int idx = itMET - mets->begin();
    edm::RefToBase<METType> metsRef = mets->refAt(idx);
    MET amet(metsRef);
    // add the generated MET
    if (addGenMET_) amet.setGenMET((*genMETs)[idx]);
    // matches to fired trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigPrimSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigPrimSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[metsRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          amet.addTriggerMatch(*trigPrim);
        }
      }
    }
    // add MET resolution info if demanded
    if (addResolutions_) {
      (*metResoCalc_)(amet);
    }
    // correct for muons if demanded ... never more: it's now done by JetMETCorrections
    // add the MET to the vector of METs
    patMETs->push_back(amet);
  }

  // sort MET in ET .. don't mess with this
  //  std::sort(patMETs->begin(), patMETs->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<MET> > myMETs(patMETs);
  iEvent.put(myMETs);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMETProducer);
