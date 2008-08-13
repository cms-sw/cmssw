#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

namespace TauTagTools{

  TrackRefVector filteredTracksByNumTrkHits(TrackRefVector theInitialTracks, int tkminTrackerHitsn){
    TrackRefVector filteredTracks;
    for(TrackRefVector::const_iterator iTk=theInitialTracks.begin();iTk!=theInitialTracks.end();iTk++){
      if ( (**iTk).numberOfValidHits() >= tkminTrackerHitsn )
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }

  TrackRefVector filteredTracks(TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2, Vertex pv){
    TrackRefVector filteredTracks;
    for(TrackRefVector::const_iterator iTk=theInitialTracks.begin();iTk!=theInitialTracks.end();iTk++){
      if ((**iTk).pt()>=tkminPt &&
	  (**iTk).normalizedChi2()<=tkmaxChi2 &&
	  fabs((**iTk).dxy(pv.position()))<=tkmaxipt &&
	  (**iTk).numberOfValidHits()>=tkminTrackerHitsn &&
	  (**iTk).hitPattern().numberOfValidPixelHits()>=tkminPixelHitsn)
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }
  TrackRefVector filteredTracks(TrackRefVector theInitialTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointmaxDZ,Vertex pv, double refpoint_Z){
    TrackRefVector filteredTracks;
    for(TrackRefVector::const_iterator iTk=theInitialTracks.begin();iTk!=theInitialTracks.end();iTk++){
      if(pv.isFake()) tktorefpointmaxDZ=30.;
      if ((**iTk).pt()>=tkminPt &&
	  (**iTk).normalizedChi2()<=tkmaxChi2 &&
	  fabs((**iTk).dxy(pv.position()))<=tkmaxipt &&
	  (**iTk).numberOfValidHits()>=tkminTrackerHitsn &&
	  (**iTk).hitPattern().numberOfValidPixelHits()>=tkminPixelHitsn &&
	  fabs((**iTk).dz()-refpoint_Z)<=tktorefpointmaxDZ)
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }

  PFCandidateRefVector filteredPFChargedHadrCandsByNumTrkHits(PFCandidateRefVector theInitialPFCands, int ChargedHadrCand_tkminTrackerHitsn){
    PFCandidateRefVector filteredPFChargedHadrCands;
    for(PFCandidateRefVector::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h  || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::mu || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::e){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	TrackRef PFChargedHadrCand_rectk = (**iPFCand).trackRef();

	if (!PFChargedHadrCand_rectk)continue;
	if ( (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn )
	  filteredPFChargedHadrCands.push_back(*iPFCand);
      }
    }
    return filteredPFChargedHadrCands;
  }

  PFCandidateRefVector filteredPFChargedHadrCands(PFCandidateRefVector theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2, Vertex pv){
    PFCandidateRefVector filteredPFChargedHadrCands;
    for(PFCandidateRefVector::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h  || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::mu || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::e){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	TrackRef PFChargedHadrCand_rectk = (**iPFCand).trackRef();
	

	if (!PFChargedHadrCand_rectk)continue;
	if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	    (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	    fabs((*PFChargedHadrCand_rectk).dxy(pv.position()))<=ChargedHadrCand_tkmaxipt &&
	    (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn &&
	    (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn) 
	  filteredPFChargedHadrCands.push_back(*iPFCand);
      }
    }
    return filteredPFChargedHadrCands;
  }
  PFCandidateRefVector filteredPFChargedHadrCands(PFCandidateRefVector theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointmaxDZ,Vertex pv, double refpoint_Z){
    if(pv.isFake()) ChargedHadrCand_tktorefpointmaxDZ = 30.;
    PFCandidateRefVector filteredPFChargedHadrCands;
    for(PFCandidateRefVector::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h  || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::mu || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::e){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	TrackRef PFChargedHadrCand_rectk = (**iPFCand).trackRef();
	if (!PFChargedHadrCand_rectk)continue;
	if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	    (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	    fabs((*PFChargedHadrCand_rectk).dxy(pv.position()))<=ChargedHadrCand_tkmaxipt &&
	    (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn &&
	    (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn &&
	    fabs((*PFChargedHadrCand_rectk).dz()-refpoint_Z)<=ChargedHadrCand_tktorefpointmaxDZ)
	  filteredPFChargedHadrCands.push_back(*iPFCand);
      }
    }
    return filteredPFChargedHadrCands;
  }
  
  PFCandidateRefVector filteredPFNeutrHadrCands(PFCandidateRefVector theInitialPFCands,double NeutrHadrCand_HcalclusminE){
    PFCandidateRefVector filteredPFNeutrHadrCands;
    for(PFCandidateRefVector::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h0){
	// *** Whether the neutral hadron candidate will be selected or not depends on its rec. HCAL cluster properties. 
	if ((**iPFCand).energy()>=NeutrHadrCand_HcalclusminE){
	  filteredPFNeutrHadrCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFNeutrHadrCands;
  }
  
  PFCandidateRefVector filteredPFGammaCands(PFCandidateRefVector theInitialPFCands,double GammaCand_EcalclusminE){
    PFCandidateRefVector filteredPFGammaCands;
    for(PFCandidateRefVector::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::gamma){
	// *** Whether the gamma candidate will be selected or not depends on its rec. ECAL cluster properties. 
	if ((**iPFCand).energy()>=GammaCand_EcalclusminE){
	  filteredPFGammaCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFGammaCands;
  }
  
  math::XYZPoint propagTrackECALSurfContactPoint(const MagneticField* theMagField,TrackRef theTrack){ 
    AnalyticalPropagator thefwdPropagator(theMagField,alongMomentum);
    math::XYZPoint propTrack_XYZPoint(0.,0.,0.);
    
    // get the initial Track FreeTrajectoryState - at outermost point position if possible, else at innermost point position:
    GlobalVector theTrack_initialGV(0.,0.,0.);
    GlobalPoint theTrack_initialGP(0.,0.,0.);
    if(theTrack->outerOk()){
      GlobalVector theTrack_initialoutermostGV(theTrack->outerMomentum().x(),theTrack->outerMomentum().y(),theTrack->outerMomentum().z());
      GlobalPoint theTrack_initialoutermostGP(theTrack->outerPosition().x(),theTrack->outerPosition().y(),theTrack->outerPosition().z());
      theTrack_initialGV=theTrack_initialoutermostGV;
      theTrack_initialGP=theTrack_initialoutermostGP;
    } else if(theTrack->innerOk()){
      GlobalVector theTrack_initialinnermostGV(theTrack->innerMomentum().x(),theTrack->innerMomentum().y(),theTrack->innerMomentum().z());
      GlobalPoint theTrack_initialinnermostGP(theTrack->innerPosition().x(),theTrack->innerPosition().y(),theTrack->innerPosition().z());
      theTrack_initialGV=theTrack_initialinnermostGV;
      theTrack_initialGP=theTrack_initialinnermostGP;
    } else return (propTrack_XYZPoint);
    GlobalTrajectoryParameters theTrack_initialGTPs(theTrack_initialGP,theTrack_initialGV,theTrack->charge(),&*theMagField);
    // FIX THIS !!!
    // need to convert from perigee to global or helix (curvilinear) frame
    // for now just an arbitrary matrix.
    HepSymMatrix covM_HepSM(6,1); covM_HepSM*=1e-6; // initialize to sigma=1e-3
    CartesianTrajectoryError cov_CTE(covM_HepSM);
    FreeTrajectoryState Track_initialFTS(theTrack_initialGTPs,cov_CTE);
    // ***
  
    // propagate to ECAL surface: 
    double ECALcorner_tantheta=ECALBounds::barrel_innerradius()/ECALBounds::barrel_halfLength();
    TrajectoryStateOnSurface Track_propagatedonECAL_TSOS=thefwdPropagator.propagate((Track_initialFTS),ECALBounds::barrelBound());
    if(!Track_propagatedonECAL_TSOS.isValid() || fabs(Track_propagatedonECAL_TSOS.globalParameters().position().perp()/Track_propagatedonECAL_TSOS.globalParameters().position().z())<ECALcorner_tantheta) {
      if(Track_propagatedonECAL_TSOS.isValid() && fabs(Track_propagatedonECAL_TSOS.globalParameters().position().perp()/Track_propagatedonECAL_TSOS.globalParameters().position().z())<ECALcorner_tantheta){     
	if(Track_propagatedonECAL_TSOS.globalParameters().position().eta()>0.){
	  Track_propagatedonECAL_TSOS=thefwdPropagator.propagate((Track_initialFTS),ECALBounds::positiveEndcapDisk());
	}else{ 
	  Track_propagatedonECAL_TSOS=thefwdPropagator.propagate((Track_initialFTS),ECALBounds::negativeEndcapDisk());
	}
      }
      if(!Track_propagatedonECAL_TSOS.isValid()){
	if((Track_initialFTS).position().eta()>0.){
	  Track_propagatedonECAL_TSOS=thefwdPropagator.propagate((Track_initialFTS),ECALBounds::positiveEndcapDisk());
	}else{ 
	  Track_propagatedonECAL_TSOS=thefwdPropagator.propagate((Track_initialFTS),ECALBounds::negativeEndcapDisk());
	}
      }
    }
    if(Track_propagatedonECAL_TSOS.isValid()){
      math::XYZPoint validpropTrack_XYZPoint(Track_propagatedonECAL_TSOS.globalPosition().x(),
					     Track_propagatedonECAL_TSOS.globalPosition().y(),
					     Track_propagatedonECAL_TSOS.globalPosition().z());
      propTrack_XYZPoint=validpropTrack_XYZPoint;
    }
    return (propTrack_XYZPoint);
  }
}


