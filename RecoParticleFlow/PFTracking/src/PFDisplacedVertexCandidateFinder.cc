#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexCandidateFinder.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "MagneticField/Engine/interface/MagneticField.h"

using namespace std;
using namespace reco;

//for debug only 
//#define PFLOW_DEBUG

PFDisplacedVertexCandidateFinder::PFDisplacedVertexCandidateFinder() : 
  vertexCandidates_( new PFDisplacedVertexCandidateCollection ),
  dcaCut_(1000),
  primaryVertexCut2_(0.0),
  dcaPInnerHitCut2_(1000.0),
  vertexCandidatesSize_(50),
  debug_(false) {

  TwoTrackMinimumDistance theMinimum(TwoTrackMinimumDistance::SlowMode);
  theMinimum_ = theMinimum;
}


PFDisplacedVertexCandidateFinder::~PFDisplacedVertexCandidateFinder() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFDisplacedVertexCandidateFinder - number of remaining elements: "
	<<eventTracks_.size()<<endl;
#endif
  
}


void PFDisplacedVertexCandidateFinder::setPrimaryVertex(
					       edm::Handle< reco::VertexCollection > mainVertexHandle, 
					       edm::Handle< reco::BeamSpot > beamSpotHandle){

  const math::XYZPoint beamSpot = beamSpotHandle.isValid() ? 
    math::XYZPoint(beamSpotHandle->x0(), beamSpotHandle->y0(), beamSpotHandle->z0()) : 
    math::XYZPoint(0, 0, 0);

  // The primary vertex is taken from the refitted list, 
  // if does not exist from the average offline beam spot position  
  // if does not exist (0,0,0) is used
  pvtx_ = mainVertexHandle.isValid() ? 
    math::XYZPoint(mainVertexHandle->begin()->x(), 
		   mainVertexHandle->begin()->y(), 
		   mainVertexHandle->begin()->z()) :
    beamSpot;
}

// Set the imput collection of tracks and calculate their
// trajectory parameters the Global Trajectory Parameters
void
PFDisplacedVertexCandidateFinder::setInput(const edm::Handle<TrackCollection>& trackh,
					   const MagneticField* magField) {

  magField_ = magField;
  trackMask_.clear();
  trackMask_.resize(trackh->size());
  eventTrackTrajectories_.clear();
  eventTrackTrajectories_.resize(trackh->size());

  for(unsigned i=0;i<trackMask_.size(); i++) trackMask_[i] = true; 

  eventTracks_.clear();
  if(trackh.isValid()) {
    for(unsigned i=0;i<trackh->size(); i++) {

      TrackRef tref( trackh, i); 
      TrackBaseRef tbref(tref);

      if( !isSelected( tbref ) ) {
	trackMask_[i] = false;
	continue;      
      }
      
      const Track* trk = tref.get();
      eventTracks_.push_back( tbref );
      eventTrackTrajectories_[i] = getGlobalTrajectoryParameters(trk);
    }
  }
}


// -------- Main function which find vertices -------- //

void 
PFDisplacedVertexCandidateFinder::findDisplacedVertexCandidates() {

  if (debug_) cout << "========= Start Finding Displaced Vertex Candidates =========" << endl;
  
  // The vertexCandidates have not been passed to the event, and need to be cleared
  if(vertexCandidates_.get() )vertexCandidates_->clear();
  else 
    vertexCandidates_.reset( new PFDisplacedVertexCandidateCollection );

  vertexCandidates_->reserve(vertexCandidatesSize_);
  for(IE ie = eventTracks_.begin(); 
      ie != eventTracks_.end();) {

    // Run the recursive procedure to find all tracks link together 
    // In one blob called Candidate

    PFDisplacedVertexCandidate tempVertexCandidate;

    ie = associate( eventTracks_.end() , ie, tempVertexCandidate);

    // Build remaining links in current block
    if(tempVertexCandidate.isValid()) {
      packLinks( tempVertexCandidate );
      vertexCandidates_->push_back( tempVertexCandidate );
    }
  }       

  if(debug_) cout << "========= End Finding Displaced Vertex Candidates =========" << endl;

}




PFDisplacedVertexCandidateFinder::IE 
PFDisplacedVertexCandidateFinder::associate(IE last, IE next, 
					    PFDisplacedVertexCandidate& tempVertexCandidate) {

    
#ifdef PFLOW_DEBUG
  if(debug_ ) cout<<"== Start the association procedure =="<<endl;
#endif

  if( last!= eventTracks_.end() ) {
    double dist = -1;
    GlobalPoint P(0,0,0);
    PFDisplacedVertexCandidate::VertexLinkTest linktest;
    link( (*last), (*next), dist, P, linktest); 


    if(dist<-0.5) {
#ifdef PFLOW_DEBUG
      if(debug_ ) cout<<"link failed"<<endl;
#endif
      return ++next; // association failed
    }
    else {
      // add next element to the current pflowblock
      tempVertexCandidate.addElement( (*next));  
      trackMask_[(*next).key()] = false;
#ifdef PFLOW_DEBUG   
      if(debug_ ) 
	cout<<"link parameters "
	    << " *next = " << (*next).key()
	    << " *last = " << (*last).key()
	    << "  dist = " << dist
	    << " P.x = " << P.x() 
	    << " P.y = " << P.y()
	    << " P.z = " << P.z()
	    << endl;
#endif
    }
  }
  else {
    // add next element to this eflowblock
#ifdef PFLOW_DEBUG   
    if(debug_ ) cout<<"adding to block element "
		    << (*next).key()
		    <<endl;
#endif
    tempVertexCandidate.addElement( (*next));  
    trackMask_[(*next).key()] = false;


  }

  // recursive call: associate next and other unused elements 
#ifdef PFLOW_DEBUG  
  if(debug_ ) {
    for(unsigned i=0; i<trackMask_.size(); i++) 
      cout << " Mask[" << i << "] = " << trackMask_[i];
    cout << "" << endl;
  }
#endif

  for(IE ie = eventTracks_.begin(); 
      ie != eventTracks_.end();) {
    
    if( ie == last || ie == next ) { ++ie; continue;} 

    // *ie already included to a block
    if( !trackMask_[(*ie).key()] ) { ++ie; continue;}

#ifdef PFLOW_DEBUG      
    if(debug_ ) cout<<"calling associate "
		    << (*next).key()
		    <<" & "
		    << (*ie).key()
		    <<endl;
#endif
    ie = associate(next, ie, tempVertexCandidate);
    
  }       

#ifdef PFLOW_DEBUG
  if(debug_ ) {
    cout<<"**** removing element "<<endl;
  }
#endif

  IE iteratorToNextFreeElement = eventTracks_.erase( next );

#ifdef PFLOW_DEBUG
  if(debug_ ) cout<< "== End the association procedure ==" <<endl;
#endif

  return iteratorToNextFreeElement;
}



void 
PFDisplacedVertexCandidateFinder::link(const TrackBaseRef& el1, 
				       const TrackBaseRef& el2, 
				       double& dist, GlobalPoint& P,
				       PFDisplacedVertexCandidate::VertexLinkTest& vertexLinkTest
				       ){
 
  if ( fabs(el1->eta()-el2->eta()) > 1) {dist = -1; return;}
  if ( el1->pt()>2 && el2->pt()>2 && fabs(el1->phi()-el2->phi()) > 1) {dist = -1; return;}

  const GlobalTrajectoryParameters& gt1 = eventTrackTrajectories_[el1.key()];
  const GlobalTrajectoryParameters& gt2 = eventTrackTrajectories_[el2.key()];
  

  // Closest approach
  theMinimum_.calculate(gt1,gt2);

  // Fill the parameters
  dist    = theMinimum_.distance();
  P       = theMinimum_.crossingPoint();
  
  vertexLinkTest = PFDisplacedVertexCandidate::LINKTEST_DCA; //rechit by default 


  // Check if the link test fails
  if (dist > dcaCut_) {dist = -1; return;}

  // Check if the closses approach point is too close to the primary vertex/beam pipe 
  double rho2 = P.x()*P.x()+P.y()*P.y();

  if ( rho2 < primaryVertexCut2_) {dist = -1; return;}

  // Check if the closses approach point is out of range of the tracker 

  double tob_rho_limit2 = 10000;
  double tec_z_limit = 270;

  if ( rho2 > tob_rho_limit2) {dist = -1; return;}
  if ( fabs(P.z()) > tec_z_limit) {dist = -1; return;}

  return;


  /*
  // Check if the inner hit of one of the tracks is not too far away from the vertex
  double dx = el1->innerPosition().x()-P.x();
  double dy = el1->innerPosition().y()-P.y();
  double dz = el1->innerPosition().z()-P.z();
  double dist2 = dx*dx+dy*dy+dz*dz; 

  if (dist2 > dcaPInnerHitCut2_) {
    dist = -1; 
    if (debug_) cout << "track " << el1.key() << " dist to vertex " << sqrt(dist2) << endl; 
    return;
  }

  dx = el2->innerPosition().x()-P.x();
  dy = el2->innerPosition().y()-P.y();
  dz = el2->innerPosition().z()-P.z();
  dist2 = dx*dx+dy*dy+dz*dz; 

  if (dist2 > dcaPInnerHitCut2_) {
    dist = -1; 
    if (debug_) cout << "track " << el2.key() << " dist to vertex " << sqrt(dist2) << endl; 
    return;
  }
  */
  
}


// Build up a matrix of all the links between different tracks in 
// In the Candidate
void 
PFDisplacedVertexCandidateFinder::packLinks( PFDisplacedVertexCandidate& vertexCandidate) {
  
  
  const vector < TrackBaseRef >& els = vertexCandidate.elements();
  
  //First Loop: update all link data
  for( unsigned i1=0; i1<els.size(); i1++ ) {
    for( unsigned i2=i1+1; i2<els.size(); i2++ ) {
      
      // no reflexive link
      if( i1==i2 ) continue;
      
      double dist = -1;
      GlobalPoint P(0,0,0); 
      PFDisplacedVertexCandidate::VertexLinkTest linktest; 

      link( els[i1], els[i2], dist, P, linktest);


#ifdef PFLOW_DEBUG
      if( debug_ )
	cout << "Setting link between elements " << i1 << " key " << els[i1].key() 
	     << " and " << i2 << " key " << els[i2].key() 
 	     << " of dist =" << dist << " computed from link test "
	     << linktest << endl;
#endif

      if(dist >-0.5) vertexCandidate.setLink( i1, i2, dist, P, linktest );
    }
  }

}



// --------------- TOOLS -------------- //



// This tool is a copy from VZeroFinder
GlobalTrajectoryParameters
PFDisplacedVertexCandidateFinder::getGlobalTrajectoryParameters
(const Track* track) const
{

  const GlobalPoint position(track->vx(),
			     track->vy(),
			     track->vz());
 
  const GlobalVector momentum(track->momentum().x(),
			      track->momentum().y(),
			      track->momentum().z());

  GlobalTrajectoryParameters gtp(position,momentum,
				 track->charge(),magField_);

  return gtp;
}


// This tool allow to pre-select the track for the displaced vertex finder
bool 
PFDisplacedVertexCandidateFinder::goodPtResolution( const TrackBaseRef& trackref) const {

  double nChi2 = trackref->normalizedChi2(); 
  double pt = trackref->pt();
  double dpt = trackref->ptError();
  double dxy = trackref->dxy(pvtx_);

  double pt_error = dpt/pt*100;

  if (debug_) cout << " PFDisplacedVertexFinder: PFrecTrack->Track Pt= "
		   << pt << " dPt/Pt = " << pt_error << "% nChi2 = " << nChi2 << endl;
  if (nChi2 > nChi2_max_ || pt < pt_min_){

    if (debug_) cout << " PFBlockAlgo: skip badly measured or low pt track"
		     << " nChi2_cut = " << 5 
		     << " pt_cut = " << 0.2 << endl;
    return false;
  }
  //  cout << "dxy = " << dxy << endl; 
  if (fabs(dxy) < dxy_ && pt < pt_min_prim_) return false;
  //  if (fabs(dxy) < 0.2 && pt < 0.8) return false;

  

  return true;
}











ostream& operator<<(std::ostream& out, const PFDisplacedVertexCandidateFinder& a) {
  if(! out) return out;
  
  out<<"====== Particle Flow Block Algorithm ======= ";
  out<<endl;
  out<<"number of unassociated elements : "<<a.eventTracks_.size()<<endl;
  out<<endl;
  
  out << " Tracks selection based on " << std::endl;
  out << " pvtx_ = " << a.pvtx_ << std::endl;
  out << " fabs(dxy) < " << a.dxy_ << " and pt < "<< a.pt_min_prim_ << std::endl;
  out << " nChi2 < " << a.nChi2_max_ << " and pt < "<< a.pt_min_ << std::endl;

  out<<endl;


  for(PFDisplacedVertexCandidateFinder::IEC ie = a.eventTracks_.begin(); 
      ie != a.eventTracks_.end(); ie++) {

    double pt = (*ie).get()->pt(); 

    math::XYZPoint Pi = (*ie).get()->innerPosition(); 
    math::XYZPoint Po = (*ie).get()->outerPosition(); 

    double innermost_radius = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y() + Pi.z()*Pi.z());
    double outermost_radius = sqrt(Po.x()*Po.x() + Po.y()*Po.y() + Po.z()*Po.z());
    double innermost_rho = sqrt(Pi.x()*Pi.x() + Pi.y()*Pi.y());
    double outermost_rho = sqrt(Po.x()*Po.x() + Po.y()*Po.y());
    
    out<<"ie = " << (*ie).key() 
       <<" pt = " << pt
       <<" innermost hit radius = " << innermost_radius << " rho = " << innermost_rho
       <<" outermost hit radius = " << outermost_radius << " rho = " << outermost_rho
       <<endl;
  }


  const std::unique_ptr< reco::PFDisplacedVertexCandidateCollection >& vertexCandidates
    = std::move(a.vertexCandidates()); 
    
  if(!vertexCandidates.get() ) {
    out<<"vertexCandidates already transfered"<<endl;
  }
  else {
    out<<"number of vertexCandidates : "<<vertexCandidates->size()<<endl;
    out<<endl;
 
    
    for(PFDisplacedVertexCandidateFinder::IBC ib=vertexCandidates->begin(); 
	ib != vertexCandidates->end(); ib++)
      ib->Dump();


    
  }
 
  return out;
}
