//
// $Id: PATMETProducer.cc,v 1.1 2008/03/06 09:23:10 llista Exp $
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
  addMuonCorr_    = iConfig.getParameter<bool>         ("addMuonCorrections");
  muonSrc_        = iConfig.getParameter<edm::InputTag>("muonSource");   
  
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

  // read in the muons if demanded
  edm::Handle<edm::View<MuonType> > muons;
  if (addMuonCorr_) {
    iEvent.getByLabel(muonSrc_, muons);
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
    // add MET resolution info if demanded
    if (addResolutions_) {
      (*metResoCalc_)(amet);
    }
    // correct for muons if demanded
    if (addMuonCorr_) {
      for (edm::View<MuonType>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); itMuon++) {
        amet.setP4(reco::Particle::LorentzVector(
            amet.px()-itMuon->px(),
            amet.py()-itMuon->py(),
            0,
            sqrt(pow(amet.px()-itMuon->px(), 2)+pow(amet.py()-itMuon->py(), 2))
        ));
      }
    }
    // add the MET to the vector of METs
    patMETs->push_back(amet);
  }

  // sort MET in ET
  std::sort(patMETs->begin(), patMETs->end(), eTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<MET> > myMETs(patMETs);
  iEvent.put(myMETs);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMETProducer);
