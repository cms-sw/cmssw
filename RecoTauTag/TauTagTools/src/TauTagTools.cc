#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

using namespace reco;
using std::string;

namespace TauTagTools{

   double computeDeltaR(const math::XYZVector& vec1, const math::XYZVector& vec2) 
   { 
      DeltaR<math::XYZVector> myMetricDR_;
      return myMetricDR_(vec1, vec2); 
   }
   double computeAngle(const math::XYZVector& vec1, const math::XYZVector& vec2)  
   {  
      Angle<math::XYZVector> myMetricAngle_;
      return myMetricAngle_(vec1, vec2); 
   }
TFormula computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage){
  //--- check functional form 
  //    given as configuration parameter for matching and signal cone sizes;
  //
  //    The size of a cone may depend on the energy "E" and/or transverse energy "ET" of the tau-jet candidate.
  //    Alternatively one can additionally use "JetOpeningDR", which specifies the opening angle of the seed jet.
  //    Any functional form that is supported by ROOT's TFormula class can be used (e.g. "3.0/E", "0.25/sqrt(ET)")
  //
  //    replace "E"  by TFormula variable "x"
  //            "ET"                      "y"
  TFormula ConeSizeTFormula;
  string ConeSizeFormulaStr = ConeSizeFormula;
  replaceSubStr(ConeSizeFormulaStr,"JetOpeningDR", "z");
  replaceSubStr(ConeSizeFormulaStr,"ET","y");
  replaceSubStr(ConeSizeFormulaStr,"E","x");
  ConeSizeTFormula.SetName("ConeSize");
  ConeSizeTFormula.SetTitle(ConeSizeFormulaStr.data()); // the function definition is actually stored in the "Title" data-member of the TFormula object
  int errorFlag = ConeSizeTFormula.Compile();
  if (errorFlag!= 0) {
    throw cms::Exception("") << "\n unsupported functional Form for " << errorMessage << " " << ConeSizeFormula << std::endl
			     << "Please check that the Definition in \"" << ConeSizeTFormula.GetName() << "\" only contains the variables \"E\" or \"ET\""
			     << " and Functions that are supported by ROOT's TFormular Class." << std::endl;
  }else return ConeSizeTFormula;
}
void replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr){
  //--- protect replacement algorithm
  //    from case that oldSubStr and newSubStr are equal
  //    (nothing to be done anyway)
  if ( oldSubStr == newSubStr ) return;
  
  //--- protect replacement algorithm
  //    from case that oldSubStr contains no characters
  //    (i.e. matches everything)
  if ( oldSubStr.empty() ) return;
  
  const string::size_type lengthOldSubStr = oldSubStr.size();
  const string::size_type lengthNewSubStr = newSubStr.size();
  
  string::size_type positionPreviousMatch = 0;
  string::size_type positionNextMatch = 0;
  
  //--- consecutively replace all occurences of oldSubStr by newSubStr;
  //    keep iterating until no occurence of oldSubStr left
  while ( (positionNextMatch = s.find(oldSubStr, positionPreviousMatch)) != string::npos ) {
    s.replace(positionNextMatch, lengthOldSubStr, newSubStr);
    positionPreviousMatch = positionNextMatch + lengthNewSubStr;
  } 
}


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
	  fabs((**iTk).dz(pv.position()))<=tktorefpointmaxDZ)
	filteredTracks.push_back(*iTk);
    }
    return filteredTracks;
  }

  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCandsByNumTrkHits(std::vector<reco::PFCandidatePtr> theInitialPFCands, int ChargedHadrCand_tkminTrackerHitsn){
    std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands;
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
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

  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2, Vertex pv){
    std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands;
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
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
  std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointmaxDZ,Vertex pv, double refpoint_Z){
    if(pv.isFake()) ChargedHadrCand_tktorefpointmaxDZ = 30.;
    std::vector<reco::PFCandidatePtr> filteredPFChargedHadrCands;
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h  || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::mu || PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::e){
	// *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
	TrackRef PFChargedHadrCand_rectk = (**iPFCand).trackRef();
	if (!PFChargedHadrCand_rectk)continue;
	if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	    (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	    fabs((*PFChargedHadrCand_rectk).dxy(pv.position()))<=ChargedHadrCand_tkmaxipt &&
	    (*PFChargedHadrCand_rectk).numberOfValidHits()>=ChargedHadrCand_tkminTrackerHitsn &&
	    (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn &&
	    fabs((*PFChargedHadrCand_rectk).dz(pv.position()))<=ChargedHadrCand_tktorefpointmaxDZ)
	  filteredPFChargedHadrCands.push_back(*iPFCand);
      }
    }
    return filteredPFChargedHadrCands;
  }
  
  std::vector<reco::PFCandidatePtr> filteredPFNeutrHadrCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double NeutrHadrCand_HcalclusMinEt){
    std::vector<reco::PFCandidatePtr> filteredPFNeutrHadrCands;
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::h0){
	// *** Whether the neutral hadron candidate will be selected or not depends on its rec. HCAL cluster properties. 
	if ((**iPFCand).et()>=NeutrHadrCand_HcalclusMinEt){
	  filteredPFNeutrHadrCands.push_back(*iPFCand);
	}
      }
    }
    return filteredPFNeutrHadrCands;
  }
  
  std::vector<reco::PFCandidatePtr> filteredPFGammaCands(std::vector<reco::PFCandidatePtr> theInitialPFCands,double GammaCand_EcalclusMinEt){
    std::vector<reco::PFCandidatePtr> filteredPFGammaCands;
    for(std::vector<reco::PFCandidatePtr>::const_iterator iPFCand=theInitialPFCands.begin();iPFCand!=theInitialPFCands.end();iPFCand++){
      if (PFCandidate::ParticleType((**iPFCand).particleId())==PFCandidate::gamma){
	// *** Whether the gamma candidate will be selected or not depends on its rec. ECAL cluster properties. 
	if ((**iPFCand).et()>=GammaCand_EcalclusMinEt){
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
    AlgebraicSymMatrix66 covM=AlgebraicMatrixID(); covM *= 1e-6; // initialize to sigma=1e-3
    CartesianTrajectoryError cov_CTE(covM);
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



  void 
  sortRefVectorByPt(std::vector<reco::PFCandidatePtr>& vec)
  {

    std::vector<size_t> indices;
    indices.reserve(vec.size());
    for(unsigned int i=0;i<vec.size();++i)
      indices.push_back(i);
    
    refVectorPtSorter sorter(vec);
    std::sort(indices.begin(),indices.end(),sorter);
    
    
    std::vector<reco::PFCandidatePtr> sorted;
    sorted.reserve(vec.size());
    
    for(unsigned int i=0;i<indices.size();++i)
      sorted.push_back(vec.at(indices.at(i)));

    vec = sorted;
  }


}
