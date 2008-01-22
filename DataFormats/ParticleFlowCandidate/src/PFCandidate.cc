#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iomanip>

using namespace reco;
using namespace std;



const float PFCandidate::bigMva_ = 999;

PFCandidate::PFCandidate() : 
  particleId_( X ),
  ecalEnergy_(-1),
  hcalEnergy_(-1),
  ps1Energy_(-1),
  ps2Energy_(-1),
  flags_(0), 
  deltaP_(-1), 
  mva_e_pi_(-PFCandidate::bigMva_),
  mva_e_mu_(-PFCandidate::bigMva_),
  mva_pi_mu_(-PFCandidate::bigMva_),
  mva_nothing_gamma_(-PFCandidate::bigMva_),
  mva_nothing_nh_(-PFCandidate::bigMva_),
  mva_gamma_nh_(-PFCandidate::bigMva_)
{}


PFCandidate::PFCandidate( Charge charge, 
			  const LorentzVector & p4, 
			  ParticleType particleId, 
			  reco::PFBlockRef blockRef ) : 
  
  LeafCandidate(charge, p4), 
  particleId_(particleId), 
  blockRef_(blockRef), 
  ecalEnergy_(0),
  hcalEnergy_(0),
  ps1Energy_(-1),
  ps2Energy_(-1),
  flags_(0),
  deltaP_(-1),
  mva_e_pi_(-PFCandidate::bigMva_),
  mva_e_mu_(-PFCandidate::bigMva_),
  mva_pi_mu_(-PFCandidate::bigMva_),
  mva_nothing_gamma_(-PFCandidate::bigMva_),
  mva_nothing_nh_(-PFCandidate::bigMva_),
  mva_gamma_nh_(-PFCandidate::bigMva_)
  
  /*       ,elementIndices_(elementIndices)  */ 
{

  // proceed with various consistency checks

  // charged candidate: track ref and charge must be non null
  if(  particleId_ == h || 
       particleId_ == e || 
       particleId_ == mu ) {
    
    if( charge == 0 ) {
      string err;
      err+="Attempt to construct a charged PFCandidate with a zero charge";
      throw cms::Exception("InconsistentValue",
			   err.c_str() );
    } 
  }
  else {
    if( charge ) { 
      string err;
      err += "Attempt to construct a neutral PFCandidate ";
      err += "with a non-zero charge";
      throw cms::Exception("InconsistentValue",
			   err.c_str() );
    } 
  }
}


PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


void PFCandidate::setTrackRef(const reco::TrackRef& ref) {
  if(!charge()) {
    string err;
    err += "PFCandidate::setTrackRef: this is a neutral candidate! ";
    err += "particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;
    
    throw cms::Exception("InconsistentReference",
			 err.c_str() );
  }

  if( particleId_ == mu ) {
    
    if(  trackRef_ != muonRef_->track() ) {
      string err;
      err += "PFCandidate::setTrackRef: inconsistent track references!";
      
      throw cms::Exception("InconsistentReference",
			   err.c_str() );
    }    
  }
  trackRef_ = ref;
}


void PFCandidate::setMuonRef(const reco::MuonRef& ref) {

  if( particleId_ != mu ) {
    string err;
    err += "PFCandidate::setMuonRef: this is not a muon! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;

    throw cms::Exception("InconsistentReference",
			 err.c_str() );
  }
  else if(  trackRef_ != muonRef_->track() ) {
    string err;
    err += "PFCandidate::setMuonRef: inconsistent track references!";
    
    throw cms::Exception("InconsistentReference",
			 err.c_str() );
  }
    
  muonRef_ = ref;
}


void PFCandidate::rescaleMomentum( double rescaleFactor ) {
  LorentzVector rescaledp4 = p4();
  rescaledp4 *= rescaleFactor;
  setP4( rescaledp4 );
}


void PFCandidate::setFlag(Flags theFlag, bool value) {
  
  if(value)
    flags_ = flags_ | (1<<theFlag);
  else 
    flags_ = flags_ ^ (1<<theFlag);
}



bool PFCandidate::flag(Flags theFlag) const {

  return (flags_>>theFlag) & 1;
}




ostream& reco::operator<<(ostream& out, 
			  const PFCandidate& c ) {
  
  if(!out) return out;
  
  out<<"\tPFCandidate type: "<<c.particleId();
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  out<<setprecision(3);
  out<<" ( pT="<<setw(7)<<c.pt();
  out<<", E ="<<setw(7)<<c.energy()<<" ) ";
  out<<", iele = unknown";
  
  //   for(unsigned i=0; i<c.elementIndices_.size(); i++) {
  //     out<<c.elementIndices_[0]<<" ";
  //   }
  //   out<<endl;
  
  out<<resetiosflags(ios::right|ios::fixed);
  
  //  out<< *(c.blockRef_)<<endl;
  
  return out;
}

