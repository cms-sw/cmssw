#include "RecoMuon/L3MuonProducer/src/QuarkoniaTrackSelector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <iostream>
#include <sstream>

QuarkoniaTrackSelector::QuarkoniaTrackSelector(const edm::ParameterSet& iConfig) :
  muonTag_(iConfig.getParameter<edm::InputTag>("muonCandidates")),
  trackTag_(iConfig.getParameter<edm::InputTag>("tracks")),
  minMasses_(iConfig.getParameter< std::vector<double> >("MinMasses")),
  maxMasses_(iConfig.getParameter< std::vector<double> >("MaxMasses")),
  checkCharge_(iConfig.getParameter<bool>("checkCharge")),
  minTrackPt_(iConfig.getParameter<double>("MinTrackPt")),
  minTrackP_(iConfig.getParameter<double>("MinTrackP")),
  maxTrackEta_(iConfig.getParameter<double>("MaxTrackEta"))
{


  muonToken_ = consumes<reco::RecoChargedCandidateCollection>(muonTag_);
  trackToken_ = consumes<reco::TrackCollection>(trackTag_);


  //register your products
  produces<reco::TrackCollection>();
  //
  // verify mass windows
  //
  bool massesValid = minMasses_.size()==maxMasses_.size();
  if ( massesValid ) {
    for ( size_t i=0; i<minMasses_.size(); ++i ) {
      if ( minMasses_[i]<0 || maxMasses_[i]<0 || 
	   minMasses_[i]>maxMasses_[i] )  massesValid = false;
    }
  }
  if ( !massesValid ) {
    edm::LogError("QuarkoniaTrackSelector") << "Inconsistency in definition of mass windows, "
					    << "no track will be selected";
    minMasses_.clear();
    maxMasses_.clear();
  }

  std::ostringstream stream;
  stream << "instantiated with parameters\n"
	 << "  muonTag  = " << muonTag_ << "\n"
	 << "  trackTag = " << trackTag_ << "\n";
  stream << "  mass windows =";
  for ( size_t i=0; i<minMasses_.size(); ++i )  
    stream << " (" << minMasses_[i] << "," << maxMasses_[i] << ")";
  stream << "\n";
  stream << "  checkCharge  = " << checkCharge_ << "\n";
  stream << "  MinTrackPt = " << minTrackPt_ << "\n";
  stream << "  MinTrackP = " << minTrackP_ << "\n";
  stream << "  MaxTrackEta = " << maxTrackEta_;
  LogDebug("QuarkoniaTrackSelector") << stream.str();
}


void
QuarkoniaTrackSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  //
  // the product
  //
  std::auto_ptr<reco::TrackCollection> product(new reco::TrackCollection);
  //
  // Muons
  //
  edm::Handle<reco::RecoChargedCandidateCollection> muonHandle;
  iEvent.getByToken(muonToken_,muonHandle);
  //
  // Tracks
  //
  edm::Handle<reco::TrackCollection> trackHandle;
  iEvent.getByToken(trackToken_,trackHandle);
  //
  // Verification
  //
  if ( !muonHandle.isValid() || !trackHandle.isValid() || minMasses_.empty() ) {
    iEvent.put(product);
    return;
  }
  //
  // Debug output
  //
  if ( edm::isDebugEnabled() ) {
    std::ostringstream stream;
    stream << "\nInput muons: # / q / pt / p / eta\n";
    for ( size_t im=0; im<muonHandle->size(); ++im ) {
      const reco::RecoChargedCandidate& muon = (*muonHandle)[im];
      stream << "   " << im << " "
	     << muon.charge() << " " << muon.pt() << " "
	     << muon.p() << " " << muon.eta() << "\n";
    }
    stream << "Input tracks: # / q / pt / p / eta\n";
    for ( size_t it=0; it<trackHandle->size(); ++it ) {
      const reco::Track& track = (*trackHandle)[it];
      stream << "   " << it << " "
	     << track.charge() << " " << track.pt() << " "
	     << track.p() << " " << track.eta() << "\n";
    }
    LogDebug("QuarkoniaTrackSelector") << stream.str();
  }
  //
  // combinations
  //
//   std::ostringstream stream;
  unsigned int nQ(0);
  unsigned int nComb(0);
  std::vector<size_t> selectedTrackIndices;
  selectedTrackIndices.reserve(muonHandle->size());
  reco::Particle::LorentzVector p4Muon;
  reco::Particle::LorentzVector p4JPsi;
  // muons
  for ( size_t im=0; im<muonHandle->size(); ++im ) {
    const reco::RecoChargedCandidate& muon = (*muonHandle)[im];
    int qMuon = muon.charge();
    p4Muon = muon.p4();
    // tracks
    for ( size_t it=0; it<trackHandle->size(); ++it ) {
      const reco::Track& track = (*trackHandle)[it];
      if ( track.pt()<minTrackPt_ || track.p()<minTrackP_ ||
	   fabs(track.eta())>maxTrackEta_ )  continue;
      if ( checkCharge_ && track.charge()!=-qMuon )  continue;
      ++nQ;
      reco::Particle::LorentzVector p4Track(track.px(),track.py(),track.pz(),
					    sqrt(track.p()*track.p()+0.0111636));
      // mass windows
      double mass = (p4Muon+p4Track).mass();
//       stream << "Combined mass = " << im << " " << it 
// 	     << " " << mass 
// 	     << " phi " << track.phi() << "\n";
      for ( size_t j=0; j<minMasses_.size(); ++j ) {
	if ( mass>minMasses_[j] && mass<maxMasses_[j] ) {
	  ++nComb;
	  if ( find(selectedTrackIndices.begin(),selectedTrackIndices.end(),it)==
	       selectedTrackIndices.end() )  selectedTrackIndices.push_back(it);
// 	  stream << "... adding " << "\n"; 
	  break;
	}
      }
    }
  }
//   LogDebug("QuarkoniaTrackSelector") << stream.str();
  //
  // filling of output collection
  //
  for ( size_t i=0; i<selectedTrackIndices.size(); ++i ) 
    product->push_back((*trackHandle)[selectedTrackIndices[i]]);
  //
  // debug output
  //
  if ( edm::isDebugEnabled() ) {
    std::ostringstream stream;
    stream << "Total number of combinations = " << muonHandle->size()*trackHandle->size()
	   << " , after charge " << nQ << " , after mass " << nComb << std::endl;
    stream << "Selected " << product->size() << " tracks with # / q / pt / eta\n";
    for ( size_t i=0; i<product->size(); ++i ) {
      const reco::Track& track = (*product)[i];
      stream << "  " << i << " " << track.charge() << " "
	     << track.pt() << " " << track.eta() << "\n";
    }
    LogDebug("QuarkoniaTrackSelector") << stream.str();
  }
  //
  iEvent.put(product);
}


//define this as a plug-in
DEFINE_FWK_MODULE(QuarkoniaTrackSelector);
