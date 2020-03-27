#include "L1Trigger/TrackFindingTMTT/interface/ConverterToTTTrack.h"
#include "FWCore/Utilities/interface/Exception.h"

//=== Convert our non-persistent L1track3D object (track candidate found by Hough transform prior to fit)
//=== to the official persistent CMSSW EDM TTrack format.

namespace TMTT {


TTTrack< Ref_Phase2TrackerDigi_ > ConverterToTTTrack::makeTTTrack(const L1track3D& trk, unsigned int iPhiSec, unsigned int iEtaReg) const {

  // Get references to stubs on this track.
  std::vector<TTStubRef> ttstubrefs = this->getStubRefs(trk); 

  // Set helix parameters.
  const unsigned int nPar4 = 4; // Number of helix parameters determined by HT.

  /*
  // Create TTTrack object using these stubs. 
  TTTrack< Ref_Phase2TrackerDigi_ > track(ttstubrefs);

  // Note which (eta,phi) sector this track was reconstructed in by HT.
  track.setSector(iPhiSec);
  track.setWedge(iEtaReg);

  // Set helix parameters.
  const unsigned int nPar = trk.nHelixParam(); // Number of helix parameters determined by HT.
  // Get and store fitted track parameters 
  GlobalPoint bsPosition(-trk.d0()*sin(trk.phi0()),
			  trk.d0()*cos(trk.phi0()),
			  trk.z0()); // Point of closest approach of track to beam-line.
  track.setPOCA(bsPosition, nPar);
  float pt = trk.pt(); // pt
  track.setMomentum(
		    GlobalVector(
				 GlobalVector::Cylindrical(
							   pt,
							   trk.phi0(), // phi
							   pt*trk.tanLambda()  // pz
							   )
				 ),
		    nPar
		    );
  track.setRInv(invPtToInvR_ * trk.qOverPt(), nPar);
  track.setChi2(trk.chi2(), nPar);
  track.setStubPtConsistency(-1, nPar);
  */

  // new TTTrack constructor
  double tmp_rinv = invPtToInvR_ * trk.qOverPt();
  double tmp_phi = trk.phi0();
  double tmp_tanL = trk.tanLambda();
  double tmp_z0 = trk.z0();
  double tmp_d0 = trk.d0();
  double tmp_chi2 = -1;
  unsigned int tmp_hit = 0;
  unsigned int tmp_npar = nPar4;
  double tmp_Bfield = settings_->getBfield(); 
  TTTrack< Ref_Phase2TrackerDigi_ > track(tmp_rinv, tmp_phi, tmp_tanL, tmp_z0, tmp_d0, tmp_chi2, 0,0,0, tmp_hit, tmp_npar, tmp_Bfield);

  // set stub references
  track.setStubRefs(ttstubrefs);

  // Note which (eta,phi) sector this track was reconstructed in by HT.
  track.setPhiSector(iPhiSec);
  track.setEtaSector(iEtaReg);

  track.setStubPtConsistency(-1); // not filled.
    
  return track;
}

//=== Convert our non-persistent L1fittedTrack object (fitted track candidate)
//=== to the official persistent CMSSW EDM TTrack format.

TTTrack< Ref_Phase2TrackerDigi_ > ConverterToTTTrack::makeTTTrack(const L1fittedTrack& trk, unsigned int iPhiSec, unsigned int iEtaReg) const{

  // Check that this track is valid.
  if (! trk.accepted()) throw cms::Exception("ConverterToTTTrack ERROR: requested to convert invalid L1fittedTrack.");

  // Get references to stubs on this track.
  std::vector<TTStubRef> ttstubrefs = this->getStubRefs(trk);

  const unsigned int nPar = trk.nHelixParam(); // Number of helix parameters determined by HT.

  /*
  // Create TTTrack object using these stubs. 
  TTTrack< Ref_Phase2TrackerDigi_ > track(ttstubrefs);

  // Note which (eta,phi) sector this track was reconstructed in by HT.
  track.setSector(iPhiSec);
  track.setWedge(iEtaReg);

  // Set helix parameters.
  // Get and store fitted track parameters 
  GlobalPoint bsPosition(-trk.d0()*sin(trk.phi0()),
			  trk.d0()*cos(trk.phi0()),
			  trk.z0()); // Point of closest approach of track to beam-line.
  track.setPOCA(bsPosition, nPar);
  float pt = trk.pt(); // pt
  track.setMomentum(
		    GlobalVector(
				 GlobalVector::Cylindrical(
							   pt,
							   trk.phi0(), // phi
							   pt*trk.tanLambda()  // pz
							   )
				 ),
		    nPar
		    );
  track.setRInv(invPtToInvR_ * trk.qOverPt(), nPar);
  track.setChi2(trk.chi2(), nPar);
  track.setStubPtConsistency(-1, nPar);
  */

  // new TTTrack constructor
  double tmp_rinv = invPtToInvR_ * trk.qOverPt();
  double tmp_phi = trk.phi0();
  double tmp_tanL = trk.tanLambda();
  double tmp_z0 = trk.z0();
  double tmp_d0 = trk.d0();
  double tmp_chi2 = -1;
  unsigned int tmp_hit = 0;
  unsigned int tmp_npar = nPar;
  double tmp_Bfield = settings_->getBfield(); 
  TTTrack< Ref_Phase2TrackerDigi_ > track(tmp_rinv, tmp_phi, tmp_tanL, tmp_z0, tmp_d0, tmp_chi2, 0,0,0, tmp_hit, tmp_npar, tmp_Bfield);

  // set stub references
  track.setStubRefs(ttstubrefs);

  // Note which (eta,phi) sector this track was reconstructed in by HT.
  track.setPhiSector(iPhiSec);
  track.setEtaSector(iEtaReg);

  track.setStubPtConsistency(-1); // not filled.

  
  return track;
}
}
