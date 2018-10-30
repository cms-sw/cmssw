#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexFinder.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"

#include "PhysicsTools/RecoAlgos/plugins/KalmanVertexFitter.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TMath.h"

using namespace std;
using namespace reco;

//for debug only 
//#define PFLOW_DEBUG

PFDisplacedVertexFinder::PFDisplacedVertexFinder() : 
  displacedVertexCandidates_( nullptr),
  displacedVertices_( new PFDisplacedVertexCollection ),
  transvSize_(0.0),
  longSize_(0.0),
  primaryVertexCut_(0.0),
  tobCut_(0.0),
  tecCut_(0.0),
  minAdaptWeight_(2.0),
  debug_(false) {}


PFDisplacedVertexFinder::~PFDisplacedVertexFinder() {}

void
PFDisplacedVertexFinder::setInput(
				  const edm::Handle<reco::PFDisplacedVertexCandidateCollection>& displacedVertexCandidates) {

  if (displacedVertexCandidates.isValid()){
    displacedVertexCandidates_ = displacedVertexCandidates.product();
  } else {
    displacedVertexCandidates_ = nullptr;
  }

}


// -------- Main function which find vertices -------- //

void
PFDisplacedVertexFinder::findDisplacedVertices() {

  if (debug_) cout << "========= Start Find Displaced Vertices =========" << endl;

  // The vertexCandidates have not been passed to the event
  // So they need to be cleared is they are not empty
  if( displacedVertices_.get() ) displacedVertices_->clear();
  else 
    displacedVertices_.reset( new PFDisplacedVertexCollection );

  if (displacedVertexCandidates_ == nullptr) {
    edm::LogInfo("EmptyVertexInput")<<"displacedVertexCandidates are not set or the setInput was called with invalid vertex";
    return;
  }

  // Prepare the collections
  PFDisplacedVertexSeedCollection tempDisplacedVertexSeeds;
  tempDisplacedVertexSeeds.reserve(4*displacedVertexCandidates_->size());
  PFDisplacedVertexCollection tempDisplacedVertices;
  tempDisplacedVertices.reserve(4*displacedVertexCandidates_->size());

  if (debug_) 
    cout << "1) Parsing displacedVertexCandidates into displacedVertexSeeds" << endl;

  // 1) Parsing displacedVertexCandidates into displacedVertexSeeds which would
  // be later used for vertex fitting

  int i = -1;

  for(auto const& idvc : *displacedVertexCandidates_) {

    i++;
    if (debug_) {
      cout << "Analyse Vertex Candidate " << i << endl;
    }
    
    findSeedsFromCandidate(idvc, tempDisplacedVertexSeeds);
    
  }     

  if (debug_) cout << "2) Merging Vertex Seeds" << endl;

  // 2) Some displacedVertexSeeds coming from different displacedVertexCandidates
  // may be closed enough to be merged together. bLocked is an array which keeps the
  // information about the seeds which are desabled.
  vector<bool> bLockedSeeds; 
  bLockedSeeds.resize(tempDisplacedVertexSeeds.size());
  mergeSeeds(tempDisplacedVertexSeeds, bLockedSeeds);
  
  if (debug_) cout << "3) Fitting Vertices From Seeds" << endl;

  // 3) Fit displacedVertices from displacedVertexSeeds
  for(unsigned idv = 0; idv < tempDisplacedVertexSeeds.size(); idv++){ 
    
    if (!tempDisplacedVertexSeeds[idv].isEmpty() && !bLockedSeeds[idv]) {
      PFDisplacedVertex displacedVertex;  
      bLockedSeeds[idv] = fitVertexFromSeed(tempDisplacedVertexSeeds[idv], displacedVertex);
      if (!bLockedSeeds[idv])  tempDisplacedVertices.emplace_back(displacedVertex);
    }
  }

  if (debug_) cout << "4) Rejecting Bad Vertices and label them" << endl;
  
  // 4) Reject displaced vertices which may be considered as fakes
  vector<bool> bLocked; 
  bLocked.resize(tempDisplacedVertices.size());
  selectAndLabelVertices(tempDisplacedVertices, bLocked);
  
  if (debug_) cout << "5) Fill the Displaced Vertices" << endl;

  // 5) Fill the displacedVertex_ which would be transfered to the producer
  displacedVertices_->reserve(tempDisplacedVertices.size());

  for(unsigned idv = 0; idv < tempDisplacedVertices.size(); idv++)
    if (!bLocked[idv]) displacedVertices_->push_back(tempDisplacedVertices[idv]);

  if (debug_) cout << "========= End Find Displaced Vertices =========" << endl;


}

// -------- Different steps of the finder algorithm -------- //


void 
PFDisplacedVertexFinder::findSeedsFromCandidate(const PFDisplacedVertexCandidate& vertexCandidate, PFDisplacedVertexSeedCollection& tempDisplacedVertexSeeds){
  
  const PFDisplacedVertexCandidate::DistMap r2Map = vertexCandidate.r2Map();
  bool bNeedNewCandidate = false;

  tempDisplacedVertexSeeds.push_back( PFDisplacedVertexSeed() );

  IDVS idvc_current;

  for (PFDisplacedVertexCandidate::DistMap::const_iterator imap = r2Map.begin();
       imap != r2Map.end(); imap++){

    unsigned ie1 = (*imap).second.first;
    unsigned ie2 = (*imap).second.second;

    if (debug_) cout << "ie1 = " << ie1 << " ie2 = " << ie2 << " radius = " << sqrt((*imap).first) << endl; 

    GlobalPoint dcaPoint = vertexCandidate.dcaPoint(ie1, ie2);
    if (fabs(dcaPoint.x()) > 1e9) continue;

    bNeedNewCandidate = true;
    for (idvc_current = tempDisplacedVertexSeeds.begin(); idvc_current != tempDisplacedVertexSeeds.end(); idvc_current++){
      if ((*idvc_current).isEmpty()) {
	bNeedNewCandidate = false;
	break;
      }
      const GlobalPoint vertexPoint = (*idvc_current).seedPoint();
      std::pair<float,float> diffs = getTransvLongDiff(vertexPoint,dcaPoint);
      if (diffs.second > longSize_) continue;
      if (diffs.first > transvSize_) continue;
      bNeedNewCandidate = false;
      break;
    }
    if (bNeedNewCandidate) {    
      if (debug_) cout << "create new displaced vertex" << endl;
      tempDisplacedVertexSeeds.push_back( PFDisplacedVertexSeed() );      
      idvc_current = tempDisplacedVertexSeeds.end();
      idvc_current--;
    }


    
    (*idvc_current).updateSeedPoint(dcaPoint, vertexCandidate.tref(ie1), vertexCandidate.tref(ie2));



  }


}




void 
PFDisplacedVertexFinder::mergeSeeds(PFDisplacedVertexSeedCollection& tempDisplacedVertexSeeds, vector<bool>& bLocked){

  // loop over displaced vertex candidates 
  // and merge them if they are close to each other
  for(unsigned idv_mother = 0;idv_mother < tempDisplacedVertexSeeds.size(); idv_mother++){
    if (!bLocked[idv_mother]){

      for (unsigned idv_daughter = idv_mother+1;idv_daughter < tempDisplacedVertexSeeds.size(); idv_daughter++){
	
	if (!bLocked[idv_daughter]){
	  if (isCloseTo(tempDisplacedVertexSeeds[idv_mother], tempDisplacedVertexSeeds[idv_daughter])) {
	
	    tempDisplacedVertexSeeds[idv_mother].mergeWith(tempDisplacedVertexSeeds[idv_daughter]);
	    bLocked[idv_daughter] = true;
	    if (debug_) cout << "Seeds " << idv_mother << " and " << idv_daughter << " merged" << endl; 
	  }
	}
      }
    }
  }

}








bool
PFDisplacedVertexFinder::fitVertexFromSeed(const PFDisplacedVertexSeed& displacedVertexSeed, PFDisplacedVertex& displacedVertex) {


  if (debug_) cout << "== Start vertexing procedure ==" << endl;


  // ---- Prepare transient track list ----

  auto const& tracksToFit = displacedVertexSeed.elements();
  const GlobalPoint& seedPoint = displacedVertexSeed.seedPoint();

  vector<TransientTrack> transTracks;
  vector<TransientTrack> transTracksRaw;
  vector<TrackBaseRef> transTracksRef;

  transTracks.reserve(tracksToFit.size());
  transTracksRaw.reserve(tracksToFit.size());
  transTracksRef.reserve(tracksToFit.size());



  TransientVertex theVertexAdaptiveRaw;
  TransientVertex theRecoVertex;

  
  // ---- 1) Clean for potentially poor seeds ------- //
  // --------------------------------------------- //

  if (tracksToFit.size() < 2) {
    if (debug_) cout << "Only one to Fit Track" << endl;
    return true;
  }

  double rho = sqrt(seedPoint.x()*seedPoint.x()+seedPoint.y()*seedPoint.y());
  double z = seedPoint.z();

  if (rho > tobCut_ || fabs(z) > tecCut_) {
    if (debug_) cout << "Seed Point out of the tracker rho = " << rho << " z = "<< z << " nTracks = " << tracksToFit.size() << endl;
    return true;
  }

  if (debug_) displacedVertexSeed.Dump();

  int nStep45 = 0;
  int nNotIterative = 0;

  // Fill vectors of TransientTracks and TrackRefs after applying preselection cuts.
  for(auto const& ie : tracksToFit){
    TransientTrack tmpTk( *(ie.get()), magField_, globTkGeomHandle_);
    transTracksRaw.emplace_back( tmpTk );
    bool nonIt = PFTrackAlgoTools::nonIterative((ie)->algo());
    bool step45 = PFTrackAlgoTools::step45((ie)->algo());
    bool highQ = PFTrackAlgoTools::highQuality((ie)->algo());   
    if (step45)
      nStep45++;
    else if (nonIt)
      nNotIterative++;
    else if (!highQ) {
      nNotIterative++;
      nStep45++;
    }

  }

  if (rho > 25 && nStep45 + nNotIterative < 1){
    if (debug_) cout << "Seed point at rho > 25 cm but no step 4-5 tracks" << endl;
    return true;
  }

  // ----------------------------------------------- //
  // ---- PRESELECT GOOD TRACKS FOR FINAL VERTEX --- //
  // ----------------------------------------------- //



  // 1)If only two track are found do not prefit


  if ( transTracksRaw.size() == 2 ){

    if (debug_) cout << "No raw fit done" << endl;
    if (switchOff2TrackVertex_) {
      if (debug_) 
	cout << "Due to probably high pile-up conditions 2 track vertices switched off" << endl;
      return true;

    }
    GlobalError globalError;

    theVertexAdaptiveRaw = TransientVertex(seedPoint,  globalError, transTracksRaw, 1.);



  } else {//branch with transTracksRaw.size of at least 3 



    if (debug_) cout << "Raw fit done." << endl;
    
    AdaptiveVertexFitter theAdaptiveFitterRaw(GeometricAnnealing(sigmacut_, t_ini_, ratio_),
					      DefaultLinearizationPointFinder(),
					      KalmanVertexUpdator<5>(), 
					      KalmanVertexTrackCompatibilityEstimator<5>(), 
					      KalmanVertexSmoother() );

    if (transTracksRaw.size() == 3){

      theVertexAdaptiveRaw = theAdaptiveFitterRaw.vertex(transTracksRaw, seedPoint);

    }
    else if ( transTracksRaw.size() < 1000){
      /// This prefit procedure allow to reduce the Warning rate from Adaptive Vertex fitter
      /// It reject also many fake tracks

      if (debug_) cout << "First test with KFT" << endl;

      KalmanVertexFitter theKalmanFitter(true);
      theVertexAdaptiveRaw = theKalmanFitter.vertex(transTracksRaw, seedPoint);
      
      if( !theVertexAdaptiveRaw.isValid() || theVertexAdaptiveRaw.totalChiSquared() < 0. ) {
	if(debug_) cout << "Prefit failed : valid? " << theVertexAdaptiveRaw.isValid() 
			<< " totalChi2 = " << theVertexAdaptiveRaw.totalChiSquared() << endl;
	return true;
      }

      if (debug_) cout << "We use KFT instead of seed point to set up a point for AVF "
		       << " x = " << theVertexAdaptiveRaw.position().x() 
		       << " y = " << theVertexAdaptiveRaw.position().y() 
		       << " z = " << theVertexAdaptiveRaw.position().z() 
		       << endl;

      // To save time: reject the Displaced vertex if it is too close to the beam pipe. 
      // Frequently it is very big vertices, with some dosens of tracks

      Vertex vtx =  theVertexAdaptiveRaw;
      rho = vtx.position().rho();

      //   cout << "primary vertex cut  = " << primaryVertexCut_ << endl;

      if (rho < primaryVertexCut_ || rho > 100) {
	if (debug_) cout << "KFT Vertex geometrically rejected with  tracks #rho = " << rho << endl;
	return true;
      }

      //     cout << "primary vertex cut  = " << primaryVertexCut_ << " rho = " << rho << endl;

      theVertexAdaptiveRaw = theAdaptiveFitterRaw.vertex(transTracksRaw, theVertexAdaptiveRaw.position());


    } else {
      edm::LogWarning("TooManyPFDVCandidates")<<"gave up vertex reco for "<< transTracksRaw.size() <<" tracks";
    } 

    if( !theVertexAdaptiveRaw.isValid() || theVertexAdaptiveRaw.totalChiSquared() < 0. ) {
      if(debug_) cout << "Fit failed : valid? " << theVertexAdaptiveRaw.isValid() 
		      << " totalChi2 = " << theVertexAdaptiveRaw.totalChiSquared() << endl;
      return true;
    }  

    // To save time: reject the Displaced vertex if it is too close to the beam pipe. 
    // Frequently it is very big vertices, with some dosens of tracks

    Vertex vtx =  theVertexAdaptiveRaw;
    rho = vtx.position().rho();

    if (rho < primaryVertexCut_)  {
      if (debug_) cout << "Vertex " << " geometrically rejected with " <<  transTracksRaw.size() << " tracks #rho = " << rho << endl;
      return true;
    }


  }


 
  // ---- Remove tracks with small weight or 
  //      big first (last) hit_to_vertex distance 
  //      and then refit                          ---- //
  

  
  for (unsigned i = 0; i < transTracksRaw.size(); i++) {

    if (debug_) cout << "Raw Weight track " << i << " = " << theVertexAdaptiveRaw.trackWeight(transTracksRaw[i]) << endl;

    if (theVertexAdaptiveRaw.trackWeight(transTracksRaw[i]) > minAdaptWeight_){

      PFTrackHitFullInfo pattern = hitPattern_.analyze(tkerTopo_, tkerGeom_, tracksToFit[i], theVertexAdaptiveRaw);

      PFDisplacedVertex::VertexTrackType vertexTrackType = getVertexTrackType(pattern);

      if (vertexTrackType != PFDisplacedVertex::T_NOT_FROM_VERTEX){

	bool bGoodTrack = helper_.isTrackSelected(transTracksRaw[i].track(), vertexTrackType);

	if (bGoodTrack){
	  transTracks.push_back(transTracksRaw[i]);
	  transTracksRef.push_back(tracksToFit[i]);
	} else {
	  if (debug_)
	    cout << "Track rejected nChi2 = " << transTracksRaw[i].track().normalizedChi2()
		 << " pt = " <<  transTracksRaw[i].track().pt()
		 << " dxy (wrt (0,0,0)) = " << transTracksRaw[i].track().dxy()
		 << " nHits = " << transTracksRaw[i].track().numberOfValidHits()
		 << " nOuterHits = " << transTracksRaw[i].track().hitPattern().numberOfLostHits(HitPattern::MISSING_OUTER_HITS) << endl;
	} 
      } else {
	
	if (debug_){ 
	  cout << "Remove track because too far away from the vertex:" << endl;
	}
	
      }
      
    }
   
  }



  if (debug_) cout << "All Tracks " << transTracksRaw.size() 
		   << " with good weight " << transTracks.size() << endl;


  // ---- Refit ---- //
  FitterType vtxFitter = F_NOTDEFINED;

  if (transTracks.size() < 2) return true;
  else if (transTracks.size() == 2){
    
    if (switchOff2TrackVertex_) {
      if (debug_) 
	cout << "Due to probably high pile-up conditions 2 track vertices switched off" << endl;
      return true;
    }
    vtxFitter = F_KALMAN;
  }
  else if (transTracks.size() > 2 && transTracksRaw.size() > transTracks.size()) 
    vtxFitter = F_ADAPTIVE;
  else if (transTracks.size() > 2 && transTracksRaw.size() == transTracks.size())  
    vtxFitter = F_DONOTREFIT;
  else return true;

  if (debug_) cout << "Vertex Fitter " << vtxFitter << endl;

  if(vtxFitter == F_KALMAN){

    KalmanVertexFitter theKalmanFitter(true);
    theRecoVertex = theKalmanFitter.vertex(transTracks, seedPoint);

  } else if(vtxFitter == F_ADAPTIVE){

    AdaptiveVertexFitter theAdaptiveFitter( 
					 GeometricAnnealing(sigmacut_, t_ini_, ratio_),
					 DefaultLinearizationPointFinder(),
					 KalmanVertexUpdator<5>(), 
					 KalmanVertexTrackCompatibilityEstimator<5>(), 
					 KalmanVertexSmoother() );
    
    theRecoVertex = theAdaptiveFitter.vertex(transTracks, seedPoint);

  } else if (vtxFitter == F_DONOTREFIT) {
    theRecoVertex = theVertexAdaptiveRaw;
  } else {
    return true;
  }


  // ---- Check if the fitted vertex is valid ---- //

  if( !theRecoVertex.isValid() || theRecoVertex.totalChiSquared() < 0. ) {
    if (debug_) cout << "Refit failed : valid? " << theRecoVertex.isValid() 
	 << " totalChi2 = " << theRecoVertex.totalChiSquared() << endl;
    return true;
  }

  // ---- Create vertex ---- //

  Vertex theRecoVtx = theRecoVertex;

  double chi2 = theRecoVtx.chi2();
  double ndf =  theRecoVtx.ndof();

    
  if (chi2 > TMath::ChisquareQuantile(0.95, ndf)) {
    if (debug_) 
      cout << "Rejected because of chi2 = " << chi2 << " ndf = " << ndf << " confid. level: " << TMath::ChisquareQuantile(0.95, ndf) << endl;
    return true;
  }
  


  // ---- Create and fill vector of refitted TransientTracks ---- //

  // -----------------------------------------------//
  // ---- Prepare and Fill the Displaced Vertex ----//
  // -----------------------------------------------//
  

  displacedVertex = theRecoVtx;
  displacedVertex.removeTracks();
  
  for(unsigned i = 0; i < transTracks.size();i++) {
    
    PFTrackHitFullInfo pattern = hitPattern_.analyze(tkerTopo_, tkerGeom_, transTracksRef[i], theRecoVertex);

    PFDisplacedVertex::VertexTrackType vertexTrackType = getVertexTrackType(pattern);

    Track refittedTrack;
    float weight =  theRecoVertex.trackWeight(transTracks[i]);


    if (weight < minAdaptWeight_) continue;

    try{
      refittedTrack =  theRecoVertex.refittedTrack(transTracks[i]).track();
    }catch( cms::Exception& exception ){
      continue;
    }

    if (debug_){
      cout << "Vertex Track Type = " << vertexTrackType << endl;

      cout << "nHitBeforeVertex = " << pattern.first.first 
	   << " nHitAfterVertex = " << pattern.second.first
	   << " nMissHitBeforeVertex = " << pattern.first.second
	   << " nMissHitAfterVertex = " << pattern.second.second
	   << " Weight = " << weight << endl;
    }

    displacedVertex.addElement(transTracksRef[i], 
			       refittedTrack, 
			       pattern, vertexTrackType, weight);

  }

  displacedVertex.setPrimaryDirection(helper_.primaryVertex());
  displacedVertex.calcKinematics();


  
  if (debug_) cout << "== End vertexing procedure ==" << endl;

  return false;

}






void 
PFDisplacedVertexFinder::selectAndLabelVertices(PFDisplacedVertexCollection& tempDisplacedVertices, vector <bool>& bLocked){

  if (debug_) cout << " 4.1) Reject vertices " << endl;

  for(unsigned idv = 0; idv < tempDisplacedVertices.size(); idv++){


    // ---- We accept a vertex only if it is not in TOB in the barrel 
    //      and not in TEC in the end caps
    //      and not too much before the first pixel layer            ---- //

    const float rho =  tempDisplacedVertices[idv].position().rho();
    const float z =  tempDisplacedVertices[idv].position().z();
	
    if (rho > tobCut_ || rho < primaryVertexCut_ || fabs(z) > tecCut_)  {
      if (debug_) cout << "Vertex " << idv  
		       << " geometrically rejected #rho = " << rho 
		       << " z = " << z << endl;
	 
      bLocked[idv] = true;

      continue;
    }

    unsigned nPrimary =  tempDisplacedVertices[idv].nPrimaryTracks();
    unsigned nMerged =  tempDisplacedVertices[idv].nMergedTracks();
    unsigned nSecondary =  tempDisplacedVertices[idv].nSecondaryTracks();      

    if (nPrimary + nMerged > 1) {
      bLocked[idv] = true;
      if (debug_) cout << "Vertex " << idv
		       << " rejected because two primary or merged tracks" << endl;
      

    }

    if (nPrimary + nMerged + nSecondary < 2){
      bLocked[idv] = true;
      if (debug_) cout << "Vertex " << idv
		       << " rejected because only one track related to the vertex" << endl;
    }


  }

  
  if (debug_) cout << " 4.2) Check for common vertices" << endl;
  
  // ---- Among the remaining vertex we shall remove one 
  //      of those which have two common tracks          ---- //

  for(unsigned idv_mother = 0; idv_mother < tempDisplacedVertices.size(); idv_mother++){
    for(unsigned idv_daughter = idv_mother+1; 
	idv_daughter < tempDisplacedVertices.size(); idv_daughter++){

      if(!bLocked[idv_daughter] && !bLocked[idv_mother]){

	const unsigned commonTrks = commonTracks(tempDisplacedVertices[idv_daughter], tempDisplacedVertices[idv_mother]);

	if (commonTrks > 1) {

	  if (debug_) cout << "Vertices " << idv_daughter << " and " << idv_mother
			   << " has many common tracks" << endl;

	  // We keep the vertex vertex which contains the most of the tracks

	  const int mother_size = tempDisplacedVertices[idv_mother].nTracks();
	  const int daughter_size = tempDisplacedVertices[idv_daughter].nTracks();
	  
	  if (mother_size > daughter_size) bLocked[idv_daughter] = true;
	  else if (mother_size < daughter_size) bLocked[idv_mother] = true;
	  else {

	    // If they have the same size we keep the vertex with the best normalised chi2

	    const float mother_normChi2 = tempDisplacedVertices[idv_mother].normalizedChi2();
	    const float daughter_normChi2 = tempDisplacedVertices[idv_daughter].normalizedChi2();
	    if (mother_normChi2 < daughter_normChi2) bLocked[idv_daughter] = true;
	    else bLocked[idv_mother] = true;
	  }
	  
	}
      }
    }
  }
  
  for(unsigned idv = 0; idv < tempDisplacedVertices.size(); idv++)
    if ( !bLocked[idv] ) bLocked[idv] = rejectAndLabelVertex(tempDisplacedVertices[idv]);
  

}

bool
PFDisplacedVertexFinder::rejectAndLabelVertex(PFDisplacedVertex& dv){

  PFDisplacedVertex::VertexType vertexType = helper_.identifyVertex(dv);
  dv.setVertexType(vertexType);
    
  return dv.isFake();

}


/// -------- Tools -------- ///

bool 
PFDisplacedVertexFinder::isCloseTo(const PFDisplacedVertexSeed& dv1, const PFDisplacedVertexSeed& dv2) const {

  const GlobalPoint& vP1 = dv1.seedPoint();
  const GlobalPoint& vP2 = dv2.seedPoint();

  std::pair<float,float> diffs = getTransvLongDiff(vP1,vP2);
  if (diffs.second > longSize_) return false;
  if (diffs.first > transvSize_) return false;
  //  if (Delta_Long < longSize_ && Delta_Transv < transvSize_) isCloseTo = true;

  return true;

}


std::pair<float,float>
PFDisplacedVertexFinder::getTransvLongDiff(const GlobalPoint& Ref, const GlobalPoint& ToProject) const {

  const auto & vRef = Ref.basicVector();
  const auto & vToProject = ToProject.basicVector();
  float vRefMag2 = vRef.mag2();
  float oneOverMag = 1.0f/sqrt(vRefMag2);

  return std::make_pair(fabs(vRef.cross(vToProject).mag()*oneOverMag),fabs((vRef.dot(vToProject)-vRefMag2)*oneOverMag));
}


reco::PFDisplacedVertex::VertexTrackType
PFDisplacedVertexFinder::getVertexTrackType(PFTrackHitFullInfo& pairTrackHitInfo) const {

  unsigned int nHitBeforeVertex = pairTrackHitInfo.first.first;
  unsigned int nHitAfterVertex = pairTrackHitInfo.second.first;

  unsigned int nMissHitBeforeVertex = pairTrackHitInfo.first.second;
  unsigned int nMissHitAfterVertex = pairTrackHitInfo.second.second;

  // For the moment those definitions are given apriori a more general study would be useful to refine those criteria

  if (nHitBeforeVertex <= 1 && nHitAfterVertex >= 3 && nMissHitAfterVertex <= 1)
    return PFDisplacedVertex::T_FROM_VERTEX;
  else if (nHitBeforeVertex >= 3 && nHitAfterVertex <= 1 && nMissHitBeforeVertex <= 1)
    return PFDisplacedVertex::T_TO_VERTEX;
  else if ((nHitBeforeVertex >= 2 && nHitAfterVertex >= 3) 
	   || 
	   (nHitBeforeVertex >= 3 && nHitAfterVertex >= 2))
    return PFDisplacedVertex::T_MERGED;
  else 
    return PFDisplacedVertex::T_NOT_FROM_VERTEX;
}


unsigned PFDisplacedVertexFinder::commonTracks(const PFDisplacedVertex& v1, const PFDisplacedVertex& v2) const { 

  vector<Track> vt1 = v1.refittedTracks();
  vector<Track> vt2 = v2.refittedTracks();

  unsigned commonTracks = 0;

  for ( unsigned il1 = 0; il1 < vt1.size(); il1++){
    unsigned il1_idx = v1.originalTrack(vt1[il1]).key();
    
    for ( unsigned il2 = 0; il2 < vt2.size(); il2++)
      if (il1_idx == v2.originalTrack(vt2[il2]).key()) {commonTracks++; break;}
    
  }

  return commonTracks;

}

std::ostream& operator<<(std::ostream& out, const PFDisplacedVertexFinder& a) {

  if(! out) return out;
  out << setprecision(3) << setw(5) << endl;
  out << "" << endl;
  out << " ====================================== " << endl;  
  out << " ====== Displaced Vertex Finder ======= " << endl;
  out << " ====================================== " << endl;  
  out << " " << endl;  

  a.helper_.Dump();
  out << "" << endl 
      << " Adaptive Vertex Fitter parameters are :"<< endl 
      << " sigmacut = " << a.sigmacut_ << " T_ini = " 
      << a.t_ini_ << " ratio = " << a.ratio_ << endl << endl; 

  const std::unique_ptr< reco::PFDisplacedVertexCollection >& displacedVertices_
    = std::move(a.displacedVertices()); 


  if(!displacedVertices_.get() ) {
    out<<"displacedVertex already transfered"<<endl;
  }
  else{

    out<<"Number of displacedVertices found : "<< displacedVertices_->size()<<endl<<endl;

    int i = -1;

    for(PFDisplacedVertexFinder::IDV idv = displacedVertices_->begin(); 
	idv != displacedVertices_->end(); idv++){
      i++;
      out << i << " "; idv->Dump(); out << "" << endl;
    }
  }
 
  return out;
}
