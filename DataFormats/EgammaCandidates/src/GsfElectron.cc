#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

using namespace reco;

GsfElectron::GsfElectron()
 : mva_(0), fbrem_(0), class_(UNKNOWN) {}

GsfElectron::GsfElectron
 ( const LorentzVector & p4, int charge,
   const GsfElectronCoreRef & core,
   const TrackClusterMatching & tcm, const TrackExtrapolations & te,
   const ClosestCtfTrack & ctfInfo,
   const FiducialFlags & ff, const ShowerShape & ss, float fbrem,
   float mva
 )
 : core_(core),
   trackClusterMatching_(tcm), trackExtrapolations_(te),
   closestCtfTrack_(ctfInfo),
   fiducialFlags_(ff), showerShape_(ss),
   mva_(mva),
   fbrem_(fbrem), class_(UNKNOWN)
 {
  setCharge(charge) ;
  setP4(p4) ;
  setVertex(te.positionAtVtx) ;
  setPdgId(-11*charge) ;
  if (isEcalDriven()) corrections_.ecalEnergy = superCluster()->energy() ;
}

void GsfElectron::correctEcalEnergy( float newEnergy, float newEnergyError )
 {
  math::XYZTLorentzVectorD momentum = p4() ;
  momentum *= newEnergy/momentum.e() ;
  setP4(momentum) ;
  showerShape_.hcalDepth1OverEcal *= corrections_.ecalEnergy/newEnergy ;
  showerShape_.hcalDepth2OverEcal *= corrections_.ecalEnergy/newEnergy ;
  trackClusterMatching_.eSuperClusterOverP *= newEnergy/corrections_.ecalEnergy ;
  trackClusterMatching_.eSeedClusterOverP *= newEnergy/corrections_.ecalEnergy ;
  trackClusterMatching_.eEleClusterOverPout *= newEnergy/corrections_.ecalEnergy ;
  corrections_.ecalEnergy = newEnergy ;
  corrections_.ecalEnergyError = newEnergyError ;
  corrections_.isEcalEnergyCorrected = true ;
 }

void GsfElectron::correctMomentum
 ( const reco::Candidate::LorentzVector & momentum,
   float trackErr, float electronErr )
 {
  setP4(momentum) ;
  corrections_.trackMomentumError = trackErr ;
  corrections_.electronMomentumError = electronErr ;
  corrections_.isMomentumCorrected = true ;
 }

bool GsfElectron::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&
	   ( checkOverlap( gsfTrack(), o->gsfTrack() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}

GsfElectron * GsfElectron::clone() const {
  return new GsfElectron( * this );
}

