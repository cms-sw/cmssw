#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"

#include "TMath.h"

using namespace std;
using namespace reco;


PFDisplacedVertex::PFDisplacedVertex() : Vertex(),
					 vertexType_(ANY),
					 primaryDirection_(0,0,0)
{}

PFDisplacedVertex::PFDisplacedVertex(Vertex& v) : Vertex(v),
						  vertexType_(ANY),
						  primaryDirection_(0,0,0)
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

  size_t pos = -1;
  
  const Track refittedTrack = PFDisplacedVertex::refittedTrack(originalTrack);

  std::vector<Track> refitTrks = refittedTracks();
  for (size_t i = 0; i < refitTrks.size(); i++){
    if ( fabs(refitTrks[i].pt() - refittedTrack.pt()) < 1.e-5 ){
      pos = i;
      continue;
    }
    
  }
  //  cout << "pos = " << pos << endl;

  return pos;

}


void 
PFDisplacedVertex::setPrimaryDirection(const math::XYZPoint& pvtx){
  primaryDirection_ = math::XYZVector(position().x(), position().y(), position().z()); 
  math::XYZVector vtx(pvtx.x(), pvtx.y(), pvtx.z());

  primaryDirection_  = primaryDirection_ - vtx;
  primaryDirection_ /= (sqrt(primaryDirection_.Mag2())+1e-10);
}


std::string 
PFDisplacedVertex::nameVertexType() const {
  switch (vertexType_){
  case ANY:   return "ANY";
  case FAKE:  return "FAKE";
  case LOOPER: return "LOOPER";
  case NUCL: return "NUCL";
  case NUCL_LOOSE: return "NUCL_LOOSE";
  case NUCL_KINK: return "NUCL_KINK";
  case CONVERSION: return "CONVERSION";
  case CONVERSION_LOOSE: return "CONVERSION_LOOSE";
  case CONVERTED_BREMM: return "CONVERTED_BREMM";
  case K0_DECAY: return "K0_DECAY";
  case LAMBDA_DECAY: return "LAMBDA_DECAY";
  case LAMBDABAR_DECAY: return "LAMBDABAR_DECAY";
  case KPLUS_DECAY: return "KPLUS_DECAY";
  case KMINUS_DECAY: return "KMINUS_DECAY";
  case KPLUS_DECAY_LOOSE: return "KPLUS_DECAY_LOOSE";
  case KMINUS_DECAY_LOOSE: return "KMINUS_DECAY_LOOSE";
  case BSM_VERTEX: return "BSM_VERTEX";
  default: return "?";
  }
  return "?";
}


const math::XYZTLorentzVector 
PFDisplacedVertex::momentum(string massHypo, VertexTrackType T, bool useRefitted, double mass) const {

  M_Hypo mHypo = M_CUSTOM;

  if (massHypo.find("PI")!=string::npos) mHypo = M_PION;
  else if (massHypo.find("KAON")!=string::npos) mHypo = M_KAON;
  else if (massHypo.find("LAMBDA")!=string::npos) mHypo = M_LAMBDA;
  else if (massHypo.find("MASSLESS")!=string::npos) mHypo = M_MASSLESS; 
  else if (massHypo.find("CUSTOM")!=string::npos) mHypo = M_CUSTOM;

  return momentum(mHypo, T, useRefitted, mass);

}


const math::XYZTLorentzVector 
PFDisplacedVertex::momentum(M_Hypo massHypo, VertexTrackType T, bool useRefitted, double mass) const {

  const double m2 = getMass2(massHypo, mass);



  math::XYZTLorentzVector P;

  for (size_t i = 0; i< tracksSize(); i++){
    bool bType = (trackTypes_[i]== T);
    if (T == T_TO_VERTEX || T == T_MERGED)
      bType =  (trackTypes_[i] == T_TO_VERTEX || trackTypes_[i] == T_MERGED);
 
    if ( bType ) {

      if (!useRefitted) {

	TrackBaseRef trackRef = originalTrack(refittedTracks()[i]);

	double p2 = trackRef->momentum().Mag2();
	P += math::XYZTLorentzVector (trackRef->momentum().x(),
				      trackRef->momentum().y(),
				      trackRef->momentum().z(),
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


const int 
PFDisplacedVertex::totalCharge() const {
  
  int charge = 0;

  for (size_t i = 0; i< tracksSize(); i++){
    if(trackTypes_[i] == T_TO_VERTEX) charge +=  refittedTracks()[i].charge();  
    else if(trackTypes_[i] == T_FROM_VERTEX) charge -=  refittedTracks()[i].charge();  
  }

  return charge;
}


const double
PFDisplacedVertex::angle_io() const {
  math::XYZTLorentzVector momentumSec = secondaryMomentum((string) "PI", true);

  math::XYZVector p_out = momentumSec.Vect();

  math::XYZVector p_in = primaryDirection();

  if (p_in.Mag2() < 1e-10) return -1;
  return acos(p_in.Dot(p_out)/sqrt(p_in.Mag2()*p_out.Mag2()))/TMath::Pi()*180.0; 
  
}

const math::XYZVector
PFDisplacedVertex:: primaryDirection() const { 

  math::XYZTLorentzVector momentumPrim = primaryMomentum((string) "PI", true);
  math::XYZTLorentzVector momentumSec = secondaryMomentum((string) "PI", true);

  math::XYZVector p_in;

  if (( isThereKindTracks(T_TO_VERTEX) || isThereKindTracks(T_MERGED) ) &&
      momentumPrim.E() >  momentumSec.E()){
    p_in = momentumPrim.Vect()/sqrt(momentumPrim.Vect().Mag2()+1e-10);
  } else {
    p_in = primaryDirection_;
  }

  return p_in;
}


const double 
PFDisplacedVertex::getMass2(M_Hypo massHypo, double mass) const {

  // pion_mass = 0.1396 GeV
  double pion_mass2 = 0.0194;
  // k0_mass = 0.4976 GeV
  double kaon_mass2 = 0.2476;
  // lambda0_mass = 1.116 GeV
  double lambda_mass2 = 1.267;
	
  if (massHypo == M_PION) return pion_mass2;
  else if (massHypo == M_KAON) return kaon_mass2;
  else if (massHypo == M_LAMBDA) return lambda_mass2;
  else if (massHypo == M_MASSLESS) return 0;
  else if (massHypo == M_CUSTOM) return mass*mass;

  cout << "Warning: undefined mass hypothesis" << endl;
  return 0;

}

void PFDisplacedVertex::Dump( ostream& out ) const {
  if(! out ) return;

  out << "" << endl;
  out << "==================== This is a Displaced Vertex type " << 
    nameVertexType() << " ===============" << endl;

  out << " Vertex chi2 = " << chi2() << " ndf = " << ndof()<< " normalised chi2 = " << normalizedChi2()<< endl;

  out << " The vertex Fitted Position is: x = " << position().x()
      << " y = " << position().y()
      << " rho = " << position().rho() 
      << " z = " << position().z() 
      << endl;

  out<< "\t--- Structure ---  " << endl;
  out<< "Number of tracks: "  << nTracks() 
     << " nPrimary " << nPrimaryTracks()
     << " nMerged " << nMergedTracks()
     << " nSecondary " << nSecondaryTracks() << endl;
              
  vector <PFDisplacedVertex::PFTrackHitFullInfo> pattern = trackHitFullInfos();
  vector <PFDisplacedVertex::VertexTrackType> trackType = trackTypes();
  for (unsigned i = 0; i < pattern.size(); i++){
    out << "track " << i 
	<< " type = " << trackType[i]
	<< " nHit BeforeVtx = " << pattern[i].first.first 
	<< " AfterVtx = " << pattern[i].second.first
	<< " MissHit BeforeVtx = " << pattern[i].first.second
	<< " AfterVtx = " << pattern[i].second.second
	<< endl;
  }

  math::XYZTLorentzVector mom_prim = primaryMomentum((string) "PI", true);
  math::XYZTLorentzVector mom_sec = secondaryMomentum((string) "PI", true);

  // out << "Primary P:\t E " << setprecision(3) << setw(5) << mom_prim.E() 
  out << "Primary P:\t E " << mom_prim.E() 
      << "\tPt = " << mom_prim.Pt()
      << "\tPz = " << mom_prim.Pz()
      << "\tM = "  << mom_prim.M() 
      << "\tEta = " << mom_prim.Eta() 
      << "\tPhi = " << mom_prim.Phi() << endl;

  out << "Secondary P:\t E " << mom_sec.E()
      << "\tPt = " << mom_sec.Pt()
      << "\tPz = " << mom_sec.Pz()
      << "\tM = "  << mom_sec.M() 
      << "\tEta = " << mom_sec.Eta()
      << "\tPhi = " << mom_sec.Phi() << endl;

  out << " The vertex Direction is x = " << primaryDirection().x()
      << " y = " << primaryDirection().y()
      << " z = " << primaryDirection().z() 
      << " eta = " << primaryDirection().eta() 
      << " phi = " << primaryDirection().phi() << endl;

  out << " Angle_io = " << angle_io() << " deg" << endl << endl;
  
}

