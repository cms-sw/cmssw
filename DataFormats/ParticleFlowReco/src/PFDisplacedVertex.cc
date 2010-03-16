#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"

using namespace std;
using namespace reco;


PFDisplacedVertex::PFDisplacedVertex() : Vertex(),
					 vertexType_(ANY)
{}

PFDisplacedVertex::PFDisplacedVertex(Vertex& v) : Vertex(v),
						  vertexType_(ANY)
{}

void 
PFDisplacedVertex::addElement( const TrackBaseRef & r, const Track & refTrack, 
			       const PFTrackHitFullInfo& hitInfo , 
			       VertexTrackType trackType, float w ) {
  add(r, refTrack, w );
  trackTypes_.push_back(trackType);
  trackHitFullInfos_.push_back(hitInfo);
}

void 
PFDisplacedVertex::cleanTracks() {

  removeTracks();
  trackTypes_.clear();
  trackHitFullInfos_.clear();

}

    
const bool 
PFDisplacedVertex::isThereKindTracks(VertexTrackType T) const {

  vector <VertexTrackType>::const_iterator iter = 
    find (trackTypes_.begin(), trackTypes_.end(), T);
  return (iter != trackTypes_.end()) ;      

}

const int 
PFDisplacedVertex::nKindTracks(VertexTrackType T) const {

  return count ( trackTypes_.begin(), trackTypes_.end(), T);

}


const size_t 
PFDisplacedVertex::trackPosition(const reco::TrackBaseRef& originalTrack) const {
  /*
  const Track refittedTrack = PFDisplacedVertex::refittedTrack(originalTrack);


  std::vector<Track>::const_iterator it =
    find_if(refittedTracks().begin(), refittedTracks().end(), TrackEqual(refittedTrack));
  if (it==refittedTracks().end())
    throw cms::Exception("Vertex") << "Refitted track not found in list\n";
  size_t pos = it - refittedTracks().begin();
  */
  size_t pos = 0;
  return pos;

}


const math::XYZTLorentzVector 
PFDisplacedVertex::momentum(string massHypo, VertexTrackType T, bool useRefitted, double mass) const {

  const double m2 = getMass2(massHypo, mass);



  math::XYZTLorentzVector P;

  for (size_t i = 0; i< tracksSize(); i++){
    bool bType = (trackTypes_[i]== T);
    if (T == T_TO_VERTEX || T == T_MERGED)
      bType =  (trackTypes_[i] == T_TO_VERTEX || trackTypes_[i] == T_MERGED);
 
    if ( bType ) {

      if (!useRefitted) {

	TrackBaseRef trackRef = originalTrack(refittedTracks()[i]);

	double p2 = trackRef->innerMomentum().Mag2();
	P += math::XYZTLorentzVector (trackRef->innerMomentum().x(),
				      trackRef->innerMomentum().y(),
				      trackRef->innerMomentum().z(),
				      sqrt(m2 + p2));
      } else {

	//	cout << "m2 " << m2 << endl; 

	double p2 = refittedTracks()[i].momentum().Mag2();
	P += math::XYZTLorentzVector (refittedTracks()[i].momentum().x(),
				      refittedTracks()[i].momentum().y(),
				      refittedTracks()[i].momentum().z(),
				      sqrt(m2 + p2));


      }
    }
  }

  return P;    

}


const double 
PFDisplacedVertex::getMass2(string massHypo, double mass) const {

  // pion_mass = 0.1396 GeV
  double pion_mass2 = 0.0194;
  // k0_mass = 0.4976 GeV
  double kaon_mass2 = 0.2476;
  // lambda0_mass = 1.116 GeV
  double lambda_mass2 = 1.267;
	
  if (massHypo.find("PI")!=string::npos) return pion_mass2;
  else if (massHypo.find("KAON")!=string::npos) return kaon_mass2;
  else if (massHypo.find("LAMBDA")!=string::npos) return lambda_mass2;
  else if (massHypo.find("MASSLESS")!=string::npos) return 0;
  else if (massHypo.find("CUSTOM")!=string::npos) return mass*mass;

  cout << "Warning: undefined mass hypothesis" << endl;
  return 0;

}

std::ostream& operator<<( std::ostream& out, const PFDisplacedVertex& co ){
  return out;
}

