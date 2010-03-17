#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexFinder.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"

#include "PhysicsTools/RecoAlgos/plugins/KalmanVertexFitter.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "TMath.h"

using namespace std;
using namespace reco;

//for debug only 
//#define PFLOW_DEBUG

PFDisplacedVertexFinder::PFDisplacedVertexFinder() : 
  displacedVertexCandidates_( new PFDisplacedVertexCandidateCollection ),
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

  if( displacedVertexCandidates_.get() ) {
    displacedVertexCandidates_->clear();
  }
  else 
    displacedVertexCandidates_.reset( new PFDisplacedVertexCandidateCollection );


  if(displacedVertexCandidates.isValid()) {
    for(unsigned i=0;i<displacedVertexCandidates->size(); i++) {
      PFDisplacedVertexCandidateRef dvcref( displacedVertexCandidates, i); 
      displacedVertexCandidates_->push_back( (*dvcref));
    }
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

  for(IDVC idvc = displacedVertexCandidates_->begin(); 
      idvc != displacedVertexCandidates_->end(); idvc++) {

    i++;
    if (debug_) {
      cout << "Analyse Vertex Candidate " << i << endl;
    }
    
    findSeedsFromCandidate(*idvc, tempDisplacedVertexSeeds);
    
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
      //      bLockedSeeds[idv] = fitVertexFromSeed(tempDisplacedVertexSeeds[idv], "KalmanVertexFitter", displacedVertex);
      bLockedSeeds[idv] = fitVertexFromSeed(tempDisplacedVertexSeeds[idv], "AdaptiveVertexFitter", displacedVertex);
      if (!bLockedSeeds[idv])  tempDisplacedVertices.push_back(displacedVertex);
    }
  }

  if (debug_) cout << "4) Rejecting Bad Vertices" << endl;
  
  // 4) Reject displaced vertices which may be considered as fakes
  vector<bool> bLocked; 
  bLocked.resize(tempDisplacedVertices.size());
  selectVertices(tempDisplacedVertices, bLocked);
  
  if (debug_) cout << "5) Fill the Displaced Vertices" << endl;

  // 5) Fill the displacedVertex_ which would be transfered to the producer
  displacedVertices_->reserve(tempDisplacedVertices.size());

  for(unsigned idv = 0; idv < tempDisplacedVertices.size(); idv++)
    if (!bLocked[idv]) displacedVertices_->push_back(tempDisplacedVertices[idv]);

  if (debug_) cout << "========= End Find Displaced Vertices =========" << endl;


}

// -------- Different steps of the finder algorithm -------- //


void 
PFDisplacedVertexFinder::findSeedsFromCandidate(PFDisplacedVertexCandidate& vertexCandidate, PFDisplacedVertexSeedCollection& tempDisplacedVertexSeeds){
  
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
      double Delta_Long = getLongDiff(vertexPoint, dcaPoint);
      double Delta_Transv = getTransvDiff(vertexPoint, dcaPoint);
      if (Delta_Long > longSize_) continue;
      if (Delta_Transv > transvSize_) continue;
      bNeedNewCandidate = false;
      break;
    }
    if (bNeedNewCandidate) {    
      if (debug_) cout << "create new displaced vertex" << endl;
      tempDisplacedVertexSeeds.push_back( PFDisplacedVertexSeed() );      
      idvc_current = tempDisplacedVertexSeeds.end();
      idvc_current--;
      bNeedNewCandidate = false;
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
	
	if (!bLocked[idv_daughter] && isCloseTo(tempDisplacedVertexSeeds[idv_mother], tempDisplacedVertexSeeds[idv_daughter])) {
	
	  tempDisplacedVertexSeeds[idv_mother].mergeWith(tempDisplacedVertexSeeds[idv_daughter]);
	  bLocked[idv_daughter] = true;
	  if (debug_) cout << "Seeds " << idv_mother << " and " << idv_daughter << " merged" << endl; 
	
	}
      }
    }
  }

}








bool
PFDisplacedVertexFinder::fitVertexFromSeed(PFDisplacedVertexSeed& displacedVertexSeed, string vtxFitter, PFDisplacedVertex& displacedVertex) {


  if (debug_) cout << "== Start vertexing procedure ==" << endl;


  // ---- Prepare transient track list ----

  set < TrackBaseRef, PFDisplacedVertexSeed::Compare > tracksToFit = displacedVertexSeed.elements();
  GlobalPoint seedPoint = displacedVertexSeed.seedPoint();

  vector<TransientTrack> transTracks;
  vector<TransientTrack> transTracksRaw;
  vector<TrackBaseRef> transTracksRef;
  vector<TrackBaseRef> transTracksRefRaw;

  transTracks.reserve(tracksToFit.size());
  transTracksRaw.reserve(tracksToFit.size());
  transTracksRef.reserve(tracksToFit.size());
  transTracksRefRaw.reserve(tracksToFit.size());

  if (tracksToFit.size() < 2) return true;

  double rho = sqrt(seedPoint.x()*seedPoint.x()+seedPoint.y()*seedPoint.y());
  double z = seedPoint.z();

  if (rho > tobCut_ || fabs(z) > tecCut_) return true;

  // Fill vectors of TransientTracks and TrackRefs after applying preselection cuts.
  for(IEset ie = tracksToFit.begin(); ie !=  tracksToFit.end(); ie++){

    TransientTrack tmpTk( *((*ie).get()), magField_, globTkGeomHandle_);
    transTracksRaw.push_back( tmpTk );
    transTracksRefRaw.push_back( *ie );
  }

  if (transTracksRaw.size() < 2) return true;

  // ----------------------------------------------- //
  // ---- Prepare transient track list is ready ---- //
  // ----------------------------------------------- //

  // ---- Define Vertex fitters and fit ---- //

  //  AdaptiveVertexFitter theAdaptiveFitterRaw;

  double sigmacut = 6;
  double Tini = 256.;
  double ratio = 0.25;

  AdaptiveVertexFitter theAdaptiveFitterRaw(
					    GeometricAnnealing(sigmacut, Tini, ratio),
					    DefaultLinearizationPointFinder(),
					    KalmanVertexUpdator<5>(), 
					    KalmanVertexTrackCompatibilityEstimator<5>(), 
					    KalmanVertexSmoother() );


  TransientVertex theVertexAdaptiveRaw;
  
  try{
    theVertexAdaptiveRaw = theAdaptiveFitterRaw.vertex(transTracksRaw, seedPoint);
  }catch( cms::Exception& exception ){
    //    cout << exception.what() << endl;
    return true;
  }

  if( !theVertexAdaptiveRaw.isValid() || theVertexAdaptiveRaw.totalChiSquared() < 0. ) return true;
  
  // To save time: reject the Displaced vertex if it is too close to the beam pipe. 
  // Frequently it is very big vertices, with some dosens of tracks

  Vertex theVtxAdaptiveRaw = theVertexAdaptiveRaw;

  rho =  theVtxAdaptiveRaw.position().rho();

  if (rho < primaryVertexCut_)  {
    if (debug_) cout << "Vertex " << " geometrically rejected with " <<  transTracksRaw.size() << " tracks #rho = " << rho << endl;
    return true;
  }

  // ---- Remove tracks with small weight or 
  //      big first (last) hit_to_vertex distance 
  //      and then refit                          ---- //



  for (unsigned i = 0; i < transTracksRaw.size(); i++) {

    if (theVertexAdaptiveRaw.trackWeight(transTracksRaw[i]) > minAdaptWeight_){

      PFTrackHitFullInfo pattern = hitPattern_.analyze(tkerGeomHandle_, transTracksRefRaw[i], theVertexAdaptiveRaw);

      PFDisplacedVertex::VertexTrackType vertexTrackType = getVertexTrackType(pattern);

      if (vertexTrackType != PFDisplacedVertex::T_NOT_FROM_VERTEX){

	transTracks.push_back(transTracksRaw[i]);
	transTracksRef.push_back(transTracksRefRaw[i]);
      } else {

	if (debug_){ 
	  cout << "Remove track because too far away from the vertex:" << endl;
	}

      }
      
    }
    // To Do: dont forget to check consistency since you remove tracks from displaced vertex 
  }

  if (debug_) cout << "All Tracks " << transTracksRaw.size() 
		   << " with good weight " << transTracks.size() << endl;

  if (transTracks.size() < 2) return true;

  // ---- Refit ---- //

  TransientVertex theRecoVertex;

  if(vtxFitter == string("KalmanVertexFitter")){

    KalmanVertexFitter theKalmanFitter(true);
    TransientVertex theVertexKalman = theKalmanFitter.vertex(transTracks, seedPoint);
    theRecoVertex = theVertexKalman;
  } else if(vtxFitter == string("AdaptiveVertexFitter")){

    AdaptiveVertexFitter theAdaptiveFitter( 
					 GeometricAnnealing(sigmacut, Tini, ratio),
					 DefaultLinearizationPointFinder(),
					 KalmanVertexUpdator<5>(), 
					 KalmanVertexTrackCompatibilityEstimator<5>(), 
					 KalmanVertexSmoother() );

    TransientVertex theVertexAdaptive = theAdaptiveFitter.vertex(transTracksRaw, seedPoint);
    theRecoVertex = theVertexAdaptive;
  }
 

  // ---- Check if the fitted vertex is valid ---- //

  if( !theRecoVertex.isValid() || theRecoVertex.totalChiSquared() < 0. ) {
    cout << "Refit failed : valid? " << theRecoVertex.isValid() 
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

  vector<TransientTrack> transientRefittedTracks;
  if( theRecoVertex.hasRefittedTracks() ) {
    transientRefittedTracks = theRecoVertex.refittedTracks();
  }


  // -----------------------------------------------//
  // ---- Prepare and Fill the Displaced Vertex ----//
  // -----------------------------------------------//
  

  displacedVertex = (PFDisplacedVertex) theRecoVtx;
  displacedVertex.removeTracks();
  
  for(unsigned i = 0; i < transTracks.size();i++) {
    
    PFTrackHitFullInfo pattern = hitPattern_.analyze(tkerGeomHandle_, transTracksRef[i], theRecoVertex);

    PFDisplacedVertex::VertexTrackType vertexTrackType = getVertexTrackType(pattern);
    
    float weight =  theRecoVertex.trackWeight(transTracks[i]);


    if (debug_){
      cout << "Vertex Track Type = " << vertexTrackType << endl;

      cout << "nHitBeforeVertex = " << pattern.first.first 
	   << " nHitAfterVertex = " << pattern.second.first
	   << " nMissHitBeforeVertex = " << pattern.first.second
	   << " nMissHitAfterVertex = " << pattern.second.second
	   << " Weight = " << weight << endl;
    }

    displacedVertex.addElement(transTracksRef[i], 
			       theRecoVertex.refittedTrack(transTracks[i]).track(), 
			       pattern, vertexTrackType, weight);


  }
  
  if (debug_) cout << "== End vertexing procedure ==" << endl;

  return false;

}






void 
PFDisplacedVertexFinder::selectVertices(const PFDisplacedVertexCollection& tempDisplacedVertices, vector <bool>& bLocked){

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
      
    if (nPrimary + nMerged > 1) {
      bLocked[idv] = true;
      if (debug_) cout << "Vertex " << idv
		       << " rejected because two primary or merged tracks" << endl;
      

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
  

}




/// -------- Tools -------- ///

bool 
PFDisplacedVertexFinder::isCloseTo(const PFDisplacedVertexSeed& dv1, const PFDisplacedVertexSeed& dv2) const {

  bool isCloseTo = false;

  const GlobalPoint vP1 = dv1.seedPoint();
  const GlobalPoint vP2 = dv2.seedPoint();

  double Delta_Long = getLongDiff(vP1, vP2);
  double Delta_Transv = getTransvDiff(vP1, vP2);
  if (Delta_Long < longSize_ && Delta_Transv < transvSize_) isCloseTo = true;

  return isCloseTo;

}


double  
PFDisplacedVertexFinder::getLongDiff(const GlobalPoint& Ref, const GlobalPoint& ToProject) const {

  Basic3DVector<double>vRef(Ref);
  Basic3DVector<double>vToProject(ToProject);
  return fabs((vRef.dot(vToProject)-vRef.mag2())/vRef.mag());

}

double  
PFDisplacedVertexFinder::getLongProj(const GlobalPoint& Ref, const GlobalVector& ToProject) const {

  Basic3DVector<double>vRef(Ref);
  Basic3DVector<double>vToProject(ToProject);
  return (vRef.dot(vToProject))/vRef.mag();


}


double  
PFDisplacedVertexFinder::getTransvDiff(const GlobalPoint& Ref, const GlobalPoint& ToProject) const {

  Basic3DVector<double>vRef(Ref);
  Basic3DVector<double>vToProject(ToProject);
  return fabs(vRef.cross(vToProject).mag()/vRef.mag());

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
  
  out<<"====== Displaced Vertex Finder ======= ";
  out<<endl;


  const std::auto_ptr< reco::PFDisplacedVertexCollection >& displacedVertices_
    = a.displacedVertices(); 


  if(!displacedVertices_.get() ) {
    out<<"displacedVertex already transfered"<<endl;
  }
  else{

    out<<"number of displacedVertices found : "<< displacedVertices_->size()<<endl;
    out<<endl;

    for(PFDisplacedVertexFinder::IDV idv = displacedVertices_->begin(); 
	idv != displacedVertices_->end(); idv++) {
 
      out << "" << endl;
      out << "==================== This is a Displaced Vertex ===============" << endl;

      out << " Vertex chi2 = " << (*idv).chi2() << " ndf = " << (*idv).ndof()<< " normalised chi2 = " << (*idv).normalizedChi2()<< endl;

      out << " The vertex Fitted Position is: x = " << (*idv).position().x()
	  << " y = " << (*idv).position().y()
	  << " rho = " << (*idv).position().rho() 
	  << " z = " << (*idv).position().z() 
	  << endl;
  
      out<< "\t--- Structure ---  " << endl;
      out<< "Number of tracks: "  << (*idv).nTracks() 
	 << " nPrimary " << (*idv).nPrimaryTracks()
	 << " nMerged " << (*idv).nMergedTracks()
	 << " nSecondary " << (*idv).nSecondaryTracks() << endl;
              
      vector <PFDisplacedVertexFinder::PFTrackHitFullInfo> pattern = (*idv).trackHitFullInfos();
      vector <PFDisplacedVertex::VertexTrackType> trackType = (*idv).trackTypes();
      for (unsigned i = 0; i < pattern.size(); i++){
	out << "track " << i 
	     << " type = " << trackType[i]
	     << " nHit BeforeVtx = " << pattern[i].first.first 
	     << " AfterVtx = " << pattern[i].second.first
	     << " MissHit BeforeVtx = " << pattern[i].first.second
	     << " AfterVtx = " << pattern[i].second.second
	     << endl;
      }


      out << "Primary P: E " << (*idv).primaryMomentum((string) "MASSLESS", false).E() 
	   << " Pt = " << (*idv).primaryMomentum((string) "MASSLESS", false).Pt() 
	   << " Pz = " << (*idv).primaryMomentum((string) "MASSLESS", false).Pz()
	   << " M = "  << (*idv).primaryMomentum((string) "MASSLESS", false).M() << endl;

      out << "Secondary P: E " << (*idv).secondaryMomentum((string) "PI", true).E() 
	   << " Pt = " << (*idv).secondaryMomentum((string) "PI", true).Pt() 
	   << " Pz = " << (*idv).secondaryMomentum((string) "PI", true).Pz()
	   << " M = "  << (*idv).secondaryMomentum((string) "PI", true).M() << endl;
    }
  }
 
  return out;
}
