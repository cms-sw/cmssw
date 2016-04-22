#include <RecoLocalMuon/GEMSegment/plugins/trackerGEM.h>

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <DataFormats/MuonReco/interface/Muon.h>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"


#include "TLorentzVector.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"


#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <DataFormats/GeometrySurface/interface/SimpleDiskBounds.h>

trackerGEM::trackerGEM(const edm::ParameterSet& iConfig) {
  gemSegmentsToken_ = consumes<GEMSegmentCollection >(iConfig.getParameter<edm::InputTag>("gemSegmentsToken"));
  generalTracksToken_ = consumes<reco::TrackCollection >(iConfig.getParameter<edm::InputTag>("generalTracksToken"));

  maxPullXGE11_   = iConfig.getParameter<double>("maxPullXGE11");
  maxDiffXGE11_   = iConfig.getParameter<double>("maxDiffXGE11");
  maxPullYGE11_   = iConfig.getParameter<double>("maxPullYGE11");
  maxDiffYGE11_   = iConfig.getParameter<double>("maxDiffYGE11");
  maxPullXGE21_   = iConfig.getParameter<double>("maxPullXGE21");
  maxDiffXGE21_   = iConfig.getParameter<double>("maxDiffXGE21");
  maxPullYGE21_   = iConfig.getParameter<double>("maxPullYGE21");
  maxDiffYGE21_   = iConfig.getParameter<double>("maxDiffYGE21");
  maxDiffPhiDirection_ = iConfig.getParameter<double>("maxDiffPhiDirection");

  produces<std::vector<reco::Muon> >();
}

trackerGEM::~trackerGEM() {}

void trackerGEM::produce(edm::Event& ev, const edm::EventSetup& setup) {
  using namespace edm;
  using namespace reco;
  using namespace std;
  
  ESHandle<MagneticField> bField;
  setup.get<IdealMagneticFieldRecord>().get(bField);
  const SteppingHelixPropagator* ThisshProp;
  ThisshProp = new SteppingHelixPropagator(&*bField,alongMomentum);

  Handle<GEMSegmentCollection> gemSegments;
  ev.getByToken(gemSegmentsToken_,gemSegments);

  Handle<TrackCollection > generalTracks;
  ev.getByToken(generalTracksToken_,generalTracks);

  std::auto_ptr<std::vector<Muon> > muons( new std::vector<Muon> ); 

  int TrackNumber = 0;
  for (std::vector<Track>::const_iterator thisTrack = generalTracks->begin();
       thisTrack != generalTracks->end(); ++thisTrack,++TrackNumber){
    //Initializing gem plane
    //Remove later
    if (thisTrack->pt() < 1.5) continue;
    if (std::abs(thisTrack->eta()) < 1.5) continue;

    ++ntracks;
    edm::LogVerbatim("trackerGEM") << "**********************************************************"<<std::endl;
    edm::LogVerbatim("trackerGEM") << "trying match to track pt = " << thisTrack->pt()
    	      << " eta = " << thisTrack->eta()
    	      << " phi = " << thisTrack->phi()
    	      <<std::endl;
     
    reco::MuonChamberMatch* foundGE11 = findGEMSegment(*thisTrack, *gemSegments, 1, ThisshProp);
    reco::MuonChamberMatch* foundGE21 = findGEMSegment(*thisTrack, *gemSegments, 3, ThisshProp);

    if (!foundGE11 && !foundGE21) continue;
    ++nmatch;
    std::vector<reco::MuonChamberMatch> muonChamberMatches;
    if (foundGE11){
      muonChamberMatches.push_back(*foundGE11);
      ++nmatch_ge11;
    }
    if (foundGE21){
      muonChamberMatches.push_back(*foundGE21);
      ++nmatch_ge21;
    }

    TrackRef thisTrackRef(generalTracks,TrackNumber);
    	   
    // temp settting the muon to track p4
    Particle::Charge q = thisTrackRef->charge();
    Particle::LorentzVector p4(thisTrackRef->px(), thisTrackRef->py(), thisTrackRef->pz(), thisTrackRef->p());
    Particle::Point vtx(thisTrackRef->vx(),thisTrackRef->vy(), thisTrackRef->vz());

    reco::Muon MuonCandidate = reco::Muon(q, p4, vtx);

    MuonCandidate.setTrack(thisTrackRef);
    // need to make track from gem seg
    MuonCandidate.setOuterTrack(thisTrackRef);
    //MuonCandidate.setType(thisSegment->nRecHits());
    MuonCandidate.setMatches(muonChamberMatches);

    //MuonCandidate.setGlobalTrackPosAtSurface(r3FinalReco_glob);
    //MuonCandidate.setGlobalTrackMomAtSurface(p3FinalReco_glob);
    //MuonCandidate.setLocalTrackPosAtSurface(r3FinalReco);
    //MuonCandidate.setLocalTrackMomAtSurface(p3FinalReco);
    //MuonCandidate.setGlobalTrackCov(covFinalReco);
    //MuonCandidate.setLocalTrackCov(C);
    
    muons->push_back(MuonCandidate);
  }
  
  // put collection in event

  ev.put(muons);
  delete ThisshProp;
}

FreeTrajectoryState
trackerGEM::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
		   int charge, const AlgebraicSymMatrix55& cov,
		   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CurvilinearTrajectoryError tCov(cov);
  
  return cov.kRows == 5 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

FreeTrajectoryState
trackerGEM::getFTS(const GlobalVector& p3, const GlobalVector& r3, 
		   int charge, const AlgebraicSymMatrix66& cov,
		   const MagneticField* field){

  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);
  
  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars) ;
}

void trackerGEM::getFromFTS(const FreeTrajectoryState& fts,
			    GlobalVector& p3, GlobalVector& r3, 
			    int& charge, AlgebraicSymMatrix66& cov){
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  GlobalVector p3T(p3GV.x(), p3GV.y(), p3GV.z());
  GlobalVector r3T(r3GP.x(), r3GP.y(), r3GP.z());
  p3 = p3T;
  r3 = r3T;  
  // p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  // r3.set(r3GP.x(), r3GP.y(), r3GP.z());
  
  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();

}

void trackerGEM::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  iSetup.get<MuonGeometryRecord>().get(gemGeom);
  ntracks = 0; nmatch = 0; nmatch_ge11 = 0; nmatch_ge21 = 0;
  n_X_MatchFound = 0; n_Y_MatchFound = 0; n_Dir_MatchFound = 0;

}
void trackerGEM::endJob()
{
  std::cout << "ntracks  = "<< ntracks <<std::endl;
  std::cout << "eff      = "<< nmatch/ntracks <<std::endl;
  std::cout << "eff ge11 = "<< nmatch_ge11/ntracks <<std::endl;
  std::cout << "eff ge21 = "<< nmatch_ge21/ntracks <<std::endl;
  std::cout << "n_X_MatchFound    = "<< n_X_MatchFound <<std::endl;
  std::cout << "n_Y_MatchFound    = "<< n_Y_MatchFound <<std::endl;
  std::cout << "n_Dir_MatchFound  = "<< n_Dir_MatchFound <<std::endl;
  
}

reco::MuonChamberMatch* trackerGEM::findGEMSegment(const reco::Track& track, const GEMSegmentCollection& gemSegments, int station, const SteppingHelixPropagator* shPropagator)
{
  int SegmentNumber = 0;
  double ClosestDelR2 = 500.;

  const GEMSegment* matchedGEMSegment = NULL;
  
  for (auto thisSegment = gemSegments.begin(); thisSegment != gemSegments.end(); 
       ++thisSegment,++SegmentNumber){
    //GEMDetId id = thisSegment->gemDetId();
    // should be segment det ID, but not working currently
    GEMDetId id = thisSegment->specificRecHits()[0].gemId();

    if (id.station() != station) continue;
    float zSign = track.pz() > 0 ? 1.0f : -1.0f;
    if ( zSign * id.region() < 0 ) continue;
    //cout << "thisSegment->nRecHits() "<< thisSegment->nRecHits()<< endl;
    //cout << "thisSegment->specificRecHits().size() "<< thisSegment->specificRecHits().size()<< endl;
    //cout << "id.station() "<< id.station()<< endl;

    LocalPoint thisPosition(thisSegment->localPosition());
    LocalVector thisDirection(thisSegment->localDirection());

    auto chamber = gemGeom->chamber(id);
    GlobalPoint SegPos(chamber->toGlobal(thisPosition));
    GlobalVector SegDir(chamber->toGlobal(thisDirection));

    edm::LogVerbatim("trackerGEM") <<" segment = "<< id.station()
    	      <<" chamber = "<< id.chamber()
    	      <<" roll = "<< id.roll()
    	      <<" x,y,z = "<< SegPos.x()
    	      <<", "<< SegPos.y()
    	      <<", "<< SegPos.z()
      	      << std::endl;

/*
    std::cout <<" station = "<< id.station() << std::endl
              <<" chamber = "<< id.chamber() << std::endl
              <<" roll = "<< id.roll() << std::endl
              <<" Global x,y,z = "<< SegPos.x() << ", " << SegPos.y() << ", " << SegPos.z() << std::endl
              <<" Local x,y,z = "<< thisPosition.x() << ", " << thisPosition.y() << ", " << thisPosition.z() << std::endl;
*/
 
    //      if ( zSign * chamber->toGlobal(thisSegment->localPosition()).z() < 0 ) continue;
    // add in deltaR cut
      
    const float zValue = SegPos.z();

    Plane *plane = new Plane(Surface::PositionType(0,0,zValue),Surface::RotationType());

    //Getting the initial variables for propagation

    int chargeReco = track.charge(); 
    GlobalVector p3reco, r3reco;

    p3reco = GlobalVector(track.outerPx(), track.outerPy(), track.outerPz());
    r3reco = GlobalVector(track.outerX(), track.outerY(), track.outerZ());

    AlgebraicSymMatrix66 covReco;
    //This is to fill the cov matrix correctly
    AlgebraicSymMatrix55 covReco_curv;
    covReco_curv = track.outerStateCovariance();
    FreeTrajectoryState initrecostate = getFTS(p3reco, r3reco, chargeReco, covReco_curv, shPropagator->magneticField());
    getFromFTS(initrecostate, p3reco, r3reco, chargeReco, covReco);

    //Now we propagate and get the propagated variables from the propagated state
    SteppingHelixStateInfo startrecostate(initrecostate);
    SteppingHelixStateInfo lastrecostate;

    //const SteppingHelixPropagator* shPropagator = 
    //dynamic_cast<const SteppingHelixPropagator*>(&*shProp);
    // for 62XSLHC
    //lastrecostate = shPropagator->propagate(startrecostate, *plane);
    //lastrecostate = shPropagator->propagateWithPath(startrecostate, *plane);
    // for 76X
    shPropagator->propagate(startrecostate, *plane,lastrecostate);
	
    FreeTrajectoryState finalrecostate;
    lastrecostate.getFreeState(finalrecostate);

    AlgebraicSymMatrix66 covFinalReco;
    GlobalVector p3FinalReco_glob, r3FinalReco_globv;
    getFromFTS(finalrecostate, p3FinalReco_glob, r3FinalReco_globv, chargeReco, covFinalReco);

    //To transform the global propagated track to local coordinates
    GlobalPoint r3FinalReco_glob(r3FinalReco_globv.x(),r3FinalReco_globv.y(),r3FinalReco_globv.z());

    LocalPoint r3FinalReco = chamber->toLocal(r3FinalReco_glob);
    LocalVector p3FinalReco=chamber->toLocal(p3FinalReco_glob);

    //The same goes for the error
    AlgebraicMatrix thisCov(4,4,0);   
    for (int i = 1; i <=4; i++){
      for (int j = 1; j <=4; j++){
	thisCov(i,j) = thisSegment->parametersError()(i,j);
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////

    LocalTrajectoryParameters ltp(r3FinalReco,p3FinalReco,chargeReco);
    JacobianCartesianToLocal jctl(chamber->surface(),ltp);
    AlgebraicMatrix56 jacobGlbToLoc = jctl.jacobian(); 

    AlgebraicMatrix55 Ctmp =  (jacobGlbToLoc * covFinalReco) * ROOT::Math::Transpose(jacobGlbToLoc); 
    AlgebraicSymMatrix55 C;  // I couldn't find any other way, so I resort to the brute force
    for(int i=0; i<5; ++i) {
      for(int j=0; j<5; ++j) {
	C[i][j] = Ctmp[i][j]; 

      }
    }  

    Double_t sigmax = sqrt(C[3][3]+thisSegment->localPositionError().xx() );      
    Double_t sigmay = sqrt(C[4][4]+thisSegment->localPositionError().yy() );

    edm::LogVerbatim("trackerGEM") <<"=================trackerGEM==================" << "\n";
    edm::LogVerbatim("trackerGEM") <<"station = "<< id.station() << "\n"
				   <<"chamber = "<< id.chamber() << "\n"
				   <<"roll = "<< id.roll() << "\n"
				   << "track r3 global : (" << r3FinalReco_glob.x() << ", " << r3FinalReco_glob.y() << ", " << r3FinalReco_glob.z() << ")" << "\n"
				   << "track p3 global : (" << p3FinalReco_glob.x() << ", " << p3FinalReco_glob.y() << ", " << p3FinalReco_glob.z() << ")" << "\n"
				   << "track r3 local : (" << r3FinalReco.x() << ", " << r3FinalReco.y() << ", " << r3FinalReco.z() << ")" << "\n"
				   << "track p3 local : (" << p3FinalReco.x() << ", " << p3FinalReco.y() << ", " << p3FinalReco.z() << ")" << "\n"
				   << "hit r3 global : (" << SegPos.x() << ", " << SegPos.y() << ", " << SegPos.z() << ")" << "\n"
				   << "hit p3 global : (" << SegDir.x() << ", " << SegDir.y() << ", " << SegDir.z() << ")" << "\n"
				   << "hit r3 local : (" << thisPosition.x() << ", " << thisPosition.y() << ", " << thisPosition.z() << ")" << "\n"
				   << "hit p3 local : (" << thisDirection.x() << ", " << thisDirection.y() << ", " << thisDirection.z() << ")" << "\n"
				   << "sigmax2 = " << C[3][3] << ", " << thisSegment->localPositionError().xx() << "\n"
				   << "sigmay2 = " << C[4][4] << ", " << thisSegment->localPositionError().yy() << "\n";


    bool X_MatchFound = false, Y_MatchFound = false, Dir_MatchFound = false;
    
    if (station == 1){
      if ( (std::abs(thisPosition.x()-r3FinalReco.x()) < (maxPullXGE11_ * sigmax)) &&
	   (std::abs(thisPosition.x()-r3FinalReco.x()) < maxDiffXGE11_ ) ) X_MatchFound = true;
      if ( (std::abs(thisPosition.y()-r3FinalReco.y()) < (maxPullYGE11_ * sigmay)) &&
	   (std::abs(thisPosition.y()-r3FinalReco.y()) < maxDiffYGE11_ ) ) Y_MatchFound = true;
    }
    if (station == 3){
      if ( (std::abs(thisPosition.x()-r3FinalReco.x()) < (maxPullXGE21_ * sigmax)) &&
	   (std::abs(thisPosition.x()-r3FinalReco.x()) < maxDiffXGE21_ ) ) X_MatchFound = true;
      if ( (std::abs(thisPosition.y()-r3FinalReco.y()) < (maxPullYGE21_ * sigmay)) &&
	   (std::abs(thisPosition.y()-r3FinalReco.y()) < maxDiffYGE21_ ) ) Y_MatchFound = true;
    }
    double segLocalPhi = thisDirection.phi();
    //-M_PI/2;
    //if (segLocalPhi < 0) segLocalPhi += M_PI;
    
    if (p3FinalReco.unit().dot(thisDirection) > 0.9) Dir_MatchFound = true;

    if (X_MatchFound) n_X_MatchFound++;
    if (Y_MatchFound) n_Y_MatchFound++;
    if (Dir_MatchFound) n_Dir_MatchFound++;
    //std::cout << "deltaPhi = " << std::abs(reco::deltaPhi(p3FinalReco.phi(),segLocalPhi)) << std::endl;
    //std::cout << "=============> X : " << X_MatchFound << ", Y : " << Y_MatchFound << ", Phi : " << Dir_MatchFound << std::endl;

    edm::LogVerbatim("trackerGEM") <<" station = "<< station
				   <<" track phi = "<< p3FinalReco.phi() 
				   <<" seg phi = "<< segLocalPhi
				   <<" deltaPhi = "<< reco::deltaPhi(p3FinalReco.phi(),segLocalPhi)
				   << std::endl;

    edm::LogVerbatim("trackerGEM") <<" deltaX = "<< thisPosition.x()-r3FinalReco.x()
				   <<" deltaX/sigma = "<< (thisPosition.x()-r3FinalReco.x())/sigmax
				   << std::endl;
    edm::LogVerbatim("trackerGEM") <<" deltaY = "<< thisPosition.y()-r3FinalReco.y()
				   <<" deltaY/sigma = "<< (thisPosition.y()-r3FinalReco.y())/sigmay
				   << std::endl;

    for (auto rechit :thisSegment->specificRecHits()){
      GEMDetId rechitid = rechit.gemId();
      //auto rechitroll = gemGeom->etaPartition(rechitid); 
      edm::LogVerbatim("trackerGEM") <<" rec hit = "<< rechitid.station()
    		<<" chamber = "<< rechitid.chamber()
    		<<" roll = "<< rechitid.roll()
    		<<" x,y,z = "<< rechit.localPosition().x()
    		<<", "<< rechit.localPosition().y()
    		<<", "<< rechit.localPosition().z()
    		<<" layer = "<< rechitid.layer()
    		<< std::endl;

    }
      
    //Check for a Match, and if there is a match, check the delR from the segment, keeping only the closest in MuonCandidate
    if (X_MatchFound && Y_MatchFound && Dir_MatchFound) {
      GlobalPoint TkPos(r3FinalReco_globv.x(),r3FinalReco_globv.y(),r3FinalReco_globv.z());
      double thisDelR2 = reco::deltaR2(SegPos,TkPos);
      if (thisDelR2 < ClosestDelR2){
	ClosestDelR2 = thisDelR2;
	matchedGEMSegment = &(*thisSegment);
      }
    }
  }
  if (matchedGEMSegment){
    reco::MuonChamberMatch* matchedChamber = new reco::MuonChamberMatch();
    matchedChamber->id = matchedGEMSegment->specificRecHits()[0].gemId();
    matchedChamber->x = matchedGEMSegment->localPosition().x();
    matchedChamber->y = matchedGEMSegment->localPosition().y();
    matchedChamber->dXdZ = matchedGEMSegment->localDirection().z()?matchedGEMSegment->localDirection().x()/matchedGEMSegment->localDirection().z():0;
    matchedChamber->dYdZ = matchedGEMSegment->localDirection().z()?matchedGEMSegment->localDirection().y()/matchedGEMSegment->localDirection().z():0;
    matchedChamber->xErr = matchedGEMSegment->localPositionError().xx();
    matchedChamber->yErr = matchedGEMSegment->localPositionError().yy();
    // need to recheck errors
    matchedChamber->dXdZErr = matchedGEMSegment->localDirectionError().xx();
    matchedChamber->dYdZErr = matchedGEMSegment->localDirectionError().yy();
    return matchedChamber;
  }
   
  return NULL;
}
DEFINE_FWK_MODULE(trackerGEM);
