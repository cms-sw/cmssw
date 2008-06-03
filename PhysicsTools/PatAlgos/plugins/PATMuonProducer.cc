//
// $Id: PATMuonProducer.cc,v 1.5.2.1 2008/05/31 19:33:38 lowette Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"

#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;


PATMuonProducer::PATMuonProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{
  // general configurables
  muonSrc_             = iConfig.getParameter<edm::InputTag>( "muonSource" );
  embedTrack_          = iConfig.getParameter<bool>         ( "embedTrack" );
  embedStandAloneMuon_ = iConfig.getParameter<bool>         ( "embedStandAloneMuon" );
  embedCombinedMuon_   = iConfig.getParameter<bool>         ( "embedCombinedMuon" );
  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>         ( "addGenMatch" );
  genMatchSrc_   = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  // resolution configurables
  addResolutions_= iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_     = iConfig.getParameter<bool>         ( "useNNResolutions" );
  muonResoFile_  = iConfig.getParameter<std::string>  ( "muonResoFile" );
  // muon ID configurables
  addMuonID_     = iConfig.getParameter<bool>         ( "addMuonID" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(muonResoFile_).fullPath(), useNNReso_);
  }

  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
  }


  // produces vector of muons
  produces<std::vector<Muon> >();

}


PATMuonProducer::~PATMuonProducer() {
  if (addResolutions_) delete theResoCalc_;
}


void PATMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of muons from the event
  edm::Handle<edm::View<MuonType> > muons;
  iEvent.getByLabel(muonSrc_, muons);

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genMatchSrc_, genMatch);

  // loop over muons
  std::vector<Muon> * patMuons = new std::vector<Muon>();
  for (edm::View<MuonType>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon) {
    // construct the Muon from the ref -> save ref to original object
    unsigned int idx = itMuon - muons->begin();
    edm::RefToBase<MuonType> muonsRef = muons->refAt(idx);

    Muon aMuon(muonsRef);
    if (embedTrack_) aMuon.embedTrack();
    if (embedStandAloneMuon_) aMuon.embedStandAloneMuon();
    if (embedCombinedMuon_) aMuon.embedCombinedMuon();

    // store the match to the generated final state muons
    if (addGenMatch_) {
      reco::GenParticleRef genMuon = (*genMatch)[muonsRef];
      if (genMuon.isNonnull() && genMuon.isAvailable() ) {
        aMuon.setGenLepton(*genMuon);
      } // leave empty if no match found
    }
    // add resolution info
    if (addResolutions_) {
      (*theResoCalc_)(aMuon);
    }

    // add muon ID info
    if (addMuonID_) {
//      aMuon.setLeptonID((float) TMath::Prob((Float_t) itMuon->combinedMuon()->chi2(), (Int_t) itMuon->combinedMuon()->ndof()));
// no combinedMuon in fastsim
      aMuon.setLeptonID((float) TMath::Prob((Float_t) itMuon->track()->chi2(), (Int_t) itMuon->track()->ndof()));
    }

     // Isolation
    if (isolator_.enabled()) {
        isolator_.fill(*muons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            aMuon.setIsolation(it->first, it->second);
        }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        aMuon.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[muonsRef]);
    }

    // add sel to selected
    patMuons->push_back(aMuon);
  }

  // sort muons in pt
  std::sort(patMuons->begin(), patMuons->end(), pTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<Muon> > ptr(patMuons);
  iEvent.put(ptr);

  if (isolator_.enabled()) isolator_.endEvent();
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMuonProducer);
