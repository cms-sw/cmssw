//
// $Id: PATMETProducer.cc,v 1.8 2008/11/28 22:05:56 lowette Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMETProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"

#include <memory>


using namespace pat;


PATMETProducer::PATMETProducer(const edm::ParameterSet & iConfig):
  userDataHelper_ ( iConfig.getParameter<edm::ParameterSet>("userData") )
{
  // initialize the configurables
  metSrc_         = iConfig.getParameter<edm::InputTag>("metSource");
  addGenMET_      = iConfig.getParameter<bool>         ("addGenMET");
  genMETSrc_      = iConfig.getParameter<edm::InputTag>("genMETSource");
  addTrigMatch_   = iConfig.getParameter<bool>         ( "addTrigMatch" );
  trigMatchSrc_   = iConfig.getParameter<std::vector<edm::InputTag> >( "trigPrimMatch" );
  addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");

  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"));
  }

  // Check to see if the user wants to add user data
  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }

  
  // produces vector of mets
  produces<std::vector<MET> >();
}


PATMETProducer::~PATMETProducer() {
}


void PATMETProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of MET's from the event
  edm::Handle<edm::View<reco::MET> > mets;
  iEvent.getByLabel(metSrc_, mets);

  if (mets->size() != 1) throw cms::Exception("Corrupt Data") << "The input MET collection " << metSrc_.encode() << " has size " << mets->size() << " instead of 1 as it should.\n";
  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);

  // Get the vector of generated met from the event if needed
  edm::Handle<edm::View<reco::GenMET> > genMETs;
  if (addGenMET_) {
    iEvent.getByLabel(genMETSrc_, genMETs);
  }

  // loop over mets
  std::vector<MET> * patMETs = new std::vector<MET>(); 
  for (edm::View<reco::MET>::const_iterator itMET = mets->begin(); itMET != mets->end(); itMET++) {
    // construct the MET from the ref -> save ref to original object
    unsigned int idx = itMET - mets->begin();
    edm::RefToBase<reco::MET> metsRef = mets->refAt(idx);
    edm::Ptr<reco::MET> metsPtr = mets->ptrAt(idx);
    MET amet(metsRef);
    // add the generated MET
    if (addGenMET_) amet.setGenMET((*genMETs)[idx]);
    // matches to trigger primitives
    if ( addTrigMatch_ ) {
      for ( size_t i = 0; i < trigMatchSrc_.size(); ++i ) {
        edm::Handle<edm::Association<TriggerPrimitiveCollection> > trigMatch;
        iEvent.getByLabel(trigMatchSrc_[i], trigMatch);
        TriggerPrimitiveRef trigPrim = (*trigMatch)[metsRef];
        if ( trigPrim.isNonnull() && trigPrim.isAvailable() ) {
          amet.addTriggerMatch(*trigPrim);
        }
      }
    }

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( amet, metsRef );
    }


    if ( useUserData_ ) {
      userDataHelper_.add( amet, iEvent, iSetup );
    }
    

    // correct for muons if demanded... never more: it's now done by JetMETCorrections
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
