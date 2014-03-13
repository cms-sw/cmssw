/** \file ME0SegmentMatcher.cc
 *
 * \author David Nash
 */

#include <RecoMuon/MuonIdentification/plugins/ME0SegmentMatcher.h>

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <DataFormats/MuonReco/interface/ME0Muon.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "DataFormats/Math/interface/deltaR.h"


ME0SegmentMatcher::ME0SegmentMatcher(const edm::ParameterSet& pas) : iev(0){
	
  produces<std::vector<reco::ME0Muon> >();  //May have to later change this to something that makes more sense, OwnVector, RefVector, etc

}

ME0SegmentMatcher::~ME0SegmentMatcher() {}

void ME0SegmentMatcher::produce(edm::Event& ev, const edm::EventSetup& setup) {

    LogDebug("ME0SegmentMatcher") << "start producing segments for " << ++iev << "th event ";

    //Getting the objects we'll need
    using namespace edm;
    ESHandle<MagneticField> bField;
    setup.get<IdealMagneticFieldRecord>().get(bField);
    ESHandle<Propagator> shProp;
    setup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong", shProp);
   
    using namespace reco;

    Handle<std::vector<EmulatedME0Segment> > OurSegments;
    ev.getByLabel<std::vector<EmulatedME0Segment> >("me0SegmentProducer", OurSegments);

    Handle <TrackCollection > generalTracks;
    ev.getByLabel <TrackCollection> ("generalTracks", generalTracks);

    std::auto_ptr<std::vector<ME0Muon> > oc( new std::vector<ME0Muon> ); 
    std::vector<ME0Muon> TempStore; 

    int TrackNumber = 0;
    std::vector<int> TkMuonNumbers, TkIndex, TkToKeep;
    std::vector<GlobalVector>FinalTrackPosition;
    //std::cout<<"generalTracks = "<<generalTracks->size()<<std::endl;
    for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
	 thisTrack != generalTracks->end(); ++thisTrack,++TrackNumber){
      //Initializing our plane
      float zSign  = thisTrack->pz()/fabs(thisTrack->pz());
      float zValue = 560. * zSign;
      Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());
      //Getting the initial variables for propagation
      int chargeReco = thisTrack->charge(); 
      GlobalVector p3reco, r3reco;

      p3reco = GlobalVector(thisTrack->outerPx(), thisTrack->outerPy(), thisTrack->outerPz());
      r3reco = GlobalVector(thisTrack->outerX(), thisTrack->outerY(), thisTrack->outerZ());

      AlgebraicSymMatrix66 covReco;
      //This is to fill the cov matrix correctly
      AlgebraicSymMatrix55 covReco_curv;
      covReco_curv = thisTrack->outerStateCovariance();
      FreeTrajectoryState initrecostate = getFTS(p3reco, r3reco, chargeReco, covReco_curv, &*bField);
      getFromFTS(initrecostate, p3reco, r3reco, chargeReco, covReco);

      //Now we propagate and get the propagated variables from the propagated state
      SteppingHelixStateInfo startrecostate(initrecostate);
      SteppingHelixStateInfo lastrecostate;

      const SteppingHelixPropagator* ThisshProp = 
	dynamic_cast<const SteppingHelixPropagator*>(&*shProp);
	
      lastrecostate = ThisshProp->propagate(startrecostate, *plane);
	
      FreeTrajectoryState finalrecostate;
      lastrecostate.getFreeState(finalrecostate);

      AlgebraicSymMatrix66 covFinalReco;
      GlobalVector p3FinalReco, r3FinalReco;
      getFromFTS(finalrecostate, p3FinalReco, r3FinalReco, chargeReco, covFinalReco);
    
      FinalTrackPosition.push_back(r3FinalReco);
      int SegmentNumber = 0;
      for (std::vector<EmulatedME0Segment>::const_iterator thisSegment = OurSegments->begin();
	   thisSegment != OurSegments->end(); ++thisSegment,++SegmentNumber){
	//EmulatedME0Segments actually have globally initialized positions and directions, so we cast them as global points and vectors
	GlobalPoint thisPosition(thisSegment->localPosition().x(),thisSegment->localPosition().y(),thisSegment->localPosition().z());
	GlobalVector thisDirection(thisSegment->localDirection().x(),thisSegment->localDirection().y(),thisSegment->localDirection().z());
	//The same goes for the error
	AlgebraicMatrix thisCov(4,4,0);   

	for (int i = 1; i <=4; i++){
	  for (int j = 1; j <=4; j++){
	    thisCov(i,j) = thisSegment->parametersError()(i,j);
	  }
	}

	//Computing the sigma for the track
	Double_t rho_track = r3FinalReco.perp();
	Double_t phi_track = r3FinalReco.phi();

	//std::cout<<r3FinalReco.eta()<<", "<<thisTrack->eta()<<std::endl;
	Double_t drhodx_track = r3FinalReco.x()/rho_track;
	Double_t drhody_track = r3FinalReco.y()/rho_track;
	Double_t dphidx_track = -r3FinalReco.y()/(rho_track*rho_track);
	Double_t dphidy_track = r3FinalReco.x()/(rho_track*rho_track);
      
	Double_t sigmarho_track = sqrt( drhodx_track*drhodx_track*covFinalReco(0,0)+
					drhody_track*drhody_track*covFinalReco(1,1)+
					drhodx_track*drhody_track*2*covFinalReco(0,1) );
      
	Double_t sigmaphi_track = sqrt( dphidx_track*dphidx_track*covFinalReco(0,0)+
					dphidy_track*dphidy_track*covFinalReco(1,1)+
					dphidx_track*dphidy_track*2*covFinalReco(0,1) );

	//Computing the sigma for the hit
	Double_t rho_hit = thisPosition.perp();
	Double_t phi_hit = thisPosition.phi();

	Double_t drhodx_hit = thisPosition.x()/rho_hit;
	Double_t drhody_hit = thisPosition.y()/rho_hit;
	Double_t dphidx_hit = -thisPosition.y()/(rho_hit*rho_hit);
	Double_t dphidy_hit = thisPosition.x()/(rho_hit*rho_hit);
      
	Double_t sigmarho_hit = sqrt( drhodx_hit*drhodx_hit*thisCov(2,2)+
				      drhody_hit*drhody_hit*thisCov(3,3)+
				      drhodx_hit*drhody_hit*2*thisCov(2,3) );
      
	Double_t sigmaphi_hit = sqrt( dphidx_hit*dphidx_hit*thisCov(2,2)+
				      dphidy_hit*dphidy_hit*thisCov(3,3)+
				      dphidx_hit*dphidy_hit*2*thisCov(2,3) );

	//Adding the sigmas
	Double_t sigmarho = sqrt(sigmarho_track*sigmarho_track + sigmarho_hit*sigmarho_hit);
	Double_t sigmaphi = sqrt(sigmaphi_track*sigmaphi_track + sigmaphi_hit*sigmaphi_hit);

	//Checking if there is a match in rho and in phi, assuming they are pointing in the same direction

	// std::cout<<"rho_hit = "<<rho_hit<<std::endl;
	// std::cout<<"rho_track = "<<rho_track<<std::endl;
	// std::cout<<"phi_hit = "<<phi_hit<<std::endl;
	// std::cout<<"phi_track = "<<phi_track<<std::endl;

	bool R_MatchFound = false, Phi_MatchFound = false;
	//std::cout<<zSign<<", "<<thisPosition.z()<<std::endl;
	if ( zSign * thisPosition.z() > 0 ) {             
	  if ( fabs(rho_hit-rho_track) < 3.0 * sigmarho) R_MatchFound = true;
	  if ( fabs(phi_hit-phi_track) < 3.0 * sigmaphi) Phi_MatchFound = true;
	}

	if (R_MatchFound && Phi_MatchFound) {
	  //std::cout<<"FOUND ONE"<<std::endl;             
	  TrackRef thisTrackRef(generalTracks,TrackNumber);
	  EmulatedME0SegmentRef thisEmulatedME0SegmentRef(OurSegments,SegmentNumber);
	  TempStore.push_back(reco::ME0Muon(thisTrackRef,thisEmulatedME0SegmentRef));
	  TkIndex.push_back(TrackNumber);
	}
      }
    }

    for (unsigned int i = 0; i < TkIndex.size(); ++i){     //Now we construct a vector of unique TrackNumbers of tracks that have been stored
      bool AlreadyStoredInTkMuonNumbers = false;
      for (unsigned int j = 0; j < TkMuonNumbers.size(); ++j){
	if (TkMuonNumbers[j]==TkIndex[i]) AlreadyStoredInTkMuonNumbers = true;
      }
      if (!AlreadyStoredInTkMuonNumbers) TkMuonNumbers.push_back(TkIndex[i]);
    }

    for (unsigned int i = 0; i < TkMuonNumbers.size(); ++i){            //Now we loop over each TrackNumber that has been stored
      int ReferenceMuonNumber = TkMuonNumbers[i];          // The muon number of the track, starts at 0 and increments
      double RefDelR = 99999.9, ComparisonIndex = 0;
      int WhichTrackToKeep=-1;
      for (std::vector<ME0Muon>::const_iterator thisMuon = TempStore.begin();    //Now we have the second nested loop, over the ME0Muons
	   thisMuon != TempStore.end(); ++thisMuon, ++ComparisonIndex){
	
	int thisMuonNumber = TkIndex[ComparisonIndex];    //The track number of the muon we are currently looking at
	if (thisMuonNumber == ReferenceMuonNumber){        //This means we're looking at one track

	  EmulatedME0SegmentRef SegRef = thisMuon->me0segment();
	  TrackRef TkRef = thisMuon->innerTrack();
	  //Here LocalPoint is used, although the local frame and global frame coincide, hence all calculations are made in global coordinates
	  //  NOTE: Correct this when making the change to "real" EmulatedME0Segments, since these will be in real local coordinates
	  LocalPoint SegPos(SegRef->localPosition().x(),SegRef->localPosition().y(),SegRef->localPosition().z());
	  //LocalPoint TkPos(TkRef->vx(),TkRef->vy(),TkRef->vz());
	  LocalPoint TkPos(FinalTrackPosition[thisMuonNumber].x(),FinalTrackPosition[thisMuonNumber].y(),FinalTrackPosition[thisMuonNumber].z());
	  double delR = reco::deltaR(SegPos,TkPos);

	  //std::cout<<"delR = "<<delR<<std::endl;

	  if (delR < RefDelR) WhichTrackToKeep = ComparisonIndex;  //Storing a list of the vector indices of tracks to keep
	                                                           //Note: These are not the same as the "Track Numbers"
	}
      }
      if (WhichTrackToKeep != -1) TkToKeep.push_back(WhichTrackToKeep);
    }

    for (unsigned int i = 0; i < TkToKeep.size(); ++i){
      int thisKeepIndex = TkToKeep[i];
      oc->push_back(TempStore[thisKeepIndex]);    //Filling the collection
    }
  	
    // put collection in event
    ev.put(oc);

}

FreeTrajectoryState
ME0SegmentMatcher::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix55& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CurvilinearTrajectoryError tCov(cov);
  
  return cov.kRows == 5 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

FreeTrajectoryState
ME0SegmentMatcher::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
			   int charge, const AlgebraicSymMatrix66& cov,
			   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void ME0SegmentMatcher::getFromFTS(const FreeTrajectoryState& fts,
				    GlobalVector& p3, GlobalVector& r3, 
				    int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  GlobalVector p3T(p3GV.x(), p3GV.y(), p3GV.z());
  GlobalVector r3T(r3GP.x(), r3GP.y(), r3GP.z());
  p3 = p3T;
  r3 = r3T;  //Yikes, was setting this to p3T instead of r3T!?!
  // p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  // r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}


 DEFINE_FWK_MODULE(ME0SegmentMatcher);
