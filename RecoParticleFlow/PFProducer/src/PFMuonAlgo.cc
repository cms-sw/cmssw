#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

bool
PFMuonAlgo::isMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isMuon(muonRef);

}

bool
PFMuonAlgo::isLooseMuon( const reco::PFBlockElement& elt ) {

  const reco::PFBlockElementTrack* eltTrack 
    = dynamic_cast<const reco::PFBlockElementTrack*>(&elt);

  assert ( eltTrack );
  reco::MuonRef muonRef = eltTrack->muonRef();

  return isLooseMuon(muonRef);

}

bool
PFMuonAlgo::isMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isGlobalMuon() ) return false;
  if ( !muonRef->isTrackerMuon() ) return false;
  if ( !muonRef->isStandAloneMuon() ) return false;

  reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  reco::TrackRef combinedMu = muonRef->combinedMuon();
  reco::TrackRef trackerMu = muonRef->track();

  /*
  std::cout << " Global  Muon pt error " 
	    << combinedMu->ptError()/combinedMu->pt() << " " 
	    << combinedMu->pt() 
	    << std::endl; 
  std::cout << " Tracker Muon pt error " 
	    << trackerMu->ptError()/trackerMu->pt() << " " 
	    << trackerMu->pt() 
	    << std::endl;
  std::cout << " STAlone Muon pt error " 
	    << standAloneMu->ptError()/standAloneMu->pt() << " " 
	    << standAloneMu->pt() 
	    << std::endl; 
  */

  double sigmaCombined = combinedMu->ptError()/(combinedMu->pt()*combinedMu->pt());
  double sigmaTracker = trackerMu->ptError()/(trackerMu->pt()*trackerMu->pt());
  double sigmaStandAlone = standAloneMu->ptError()/(standAloneMu->pt()*standAloneMu->pt());

  bool combined = combinedMu->ptError()/combinedMu->pt() < 0.20;
  bool tracker = trackerMu->ptError()/trackerMu->pt() < 0.20;
  bool standAlone = standAloneMu->ptError()/standAloneMu->pt() < 0.20;

  double delta1 =  combined && tracker ?
    fabs(1./combinedMu->pt() -1./trackerMu->pt())
    /sqrt(sigmaCombined*sigmaCombined + sigmaTracker*sigmaTracker) : 100.; 
  double delta2 = combined && standAlone ?
    fabs(1./combinedMu->pt() -1./standAloneMu->pt())
    /sqrt(sigmaCombined*sigmaCombined + sigmaStandAlone*sigmaStandAlone) : 100.;
  double delta3 = standAlone && tracker ?
    fabs(1./standAloneMu->pt() -1./trackerMu->pt())
    /sqrt(sigmaStandAlone*sigmaStandAlone + sigmaTracker*sigmaTracker) : 100.;

  double delta = std::min(delta3,std::min(delta1,delta2));
  // std::cout << "delta = " << delta << std::endl;

  double ratio = 
    combinedMu->ptError()/combinedMu->pt()
    / (trackerMu->ptError()/trackerMu->pt());
  //if ( ratio > 2. && delta < 3. ) std::cout << "ALARM ! " << ratio << ", " << delta << std::endl;
 
  return ( combinedMu->pt() < 50. || ratio < 2. ) && delta < 3.;

}

bool
PFMuonAlgo::isLooseMuon( const reco::MuonRef& muonRef ) {

  if ( !muonRef.isNonnull() ) return false;
  if ( !muonRef->isGlobalMuon() ) return false;
  if ( !muonRef->isTrackerMuon() ) return false;
  
  reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  reco::TrackRef combinedMu = muonRef->combinedMuon();
  reco::TrackRef trackerMu = muonRef->track();

  // Some accuracy required on the momentum
  bool combined = combinedMu->ptError()/combinedMu->pt() < 0.25;
  bool tracker = trackerMu->ptError()/trackerMu->pt() < 0.25;
  // bool standAlone = standAloneMu->ptError()/standAloneMu->pt() < 0.25;

  bool combined40 = combinedMu->ptError()/combinedMu->pt() < 0.40;
  bool tracker40 = trackerMu->ptError()/trackerMu->pt() < 0.40;

  if ( !combined40 || !tracker40 ) return false;
  if ( !combined && !tracker ) return false;

  return true;

}
