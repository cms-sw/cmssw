#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
//#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>
#include <iomanip>

using namespace reco;
using namespace std;



const float PFCandidate::bigMva_ = 999;

PFCandidate::PFCandidate() : 
  particleId_( X ),
  ecalEnergy_(-1),
  hcalEnergy_(-1),
  rawEcalEnergy_(-1),
  rawHcalEnergy_(-1),
  ps1Energy_(-1),
  ps2Energy_(-1),
  flags_(0), 
  deltaP_(-1), 
  mva_e_pi_(-PFCandidate::bigMva_),
  mva_e_mu_(-PFCandidate::bigMva_),
  mva_pi_mu_(-PFCandidate::bigMva_),
  mva_nothing_gamma_(-PFCandidate::bigMva_),
  mva_nothing_nh_(-PFCandidate::bigMva_),
  mva_gamma_nh_(-PFCandidate::bigMva_) {
  
  setPdgId( translateTypeToPdgId( particleId_ ) );
}


PFCandidate::PFCandidate( const PFCandidatePtr& sourcePtr ) {
  *this = *sourcePtr;
  sourcePtr_ = sourcePtr;
}


PFCandidate::PFCandidate( Charge charge, 
			  const LorentzVector & p4, 
			  ParticleType particleId ) : 
  
  CompositeCandidate(charge, p4), 
  particleId_(particleId), 
//   blockRef_(blockRef), 
  ecalEnergy_(0),
  hcalEnergy_(0),
  rawEcalEnergy_(0),
  rawHcalEnergy_(0),
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
  setPdgId( translateTypeToPdgId( particleId_ ) );
}



PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


void PFCandidate::addElementInBlock( const reco::PFBlockRef& blockref,
                                     unsigned elementIndex ) {
  elementsInBlocks_.push_back( make_pair(blockref, elementIndex) );
}


int PFCandidate::translateTypeToPdgId( ParticleType type ) const {
  
  int thecharge = charge();

  switch( type ) {
  case h:     return thecharge*211; // pi+
  case e:     return thecharge*(-11);
  case mu:    return thecharge*(-13);
  case gamma: return 22;
  case h0:    return 130; // K_L0
  case h_HF:         return 130; // dummy pdg code 
  case egamma_HF:    return 22;  // dummy pdg code
  case X: 
  default:    return 0;  
  }
}


void PFCandidate::setParticleType( ParticleType type ) {
  particleId_ = type;
  setPdgId( translateTypeToPdgId( type ) );
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

  trackRef_ = ref;
}

void PFCandidate::setGsfTrackRef(const reco::GsfTrackRef& ref) {
  if( particleId_ != e ) {
    string err;
    err += "PFCandidate::setGsfTrackRef: this is not an electron ! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }
  gsfTrackRef_ = ref;
}

void PFCandidate::setMuonRef(const reco::MuonRef& ref) {

  if(  trackRef_ != ref->track() ) {
    string err;
    err += "PFCandidate::setMuonRef: inconsistent track references!";
    
    throw cms::Exception("InconsistentReference",
			 err.c_str() );
  }
    
  muonRef_ = ref;
}

void PFCandidate::setConversionRef(const reco::ConversionRef& ref) {

  if( particleId_ != gamma ) {
    string err;
    err += "PFCandidate::setConversionRef: this is not a (converted) photon ! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }
  else if(  !flag( GAMMA_TO_GAMMACONV ) ) {
    string err;
    err += "PFCandidate::setConversionRef: particule flag is not GAMMA_TO_GAMMACONV";

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }

  conversionRef_ = ref;
}



void PFCandidate::setDisplacedVertexRef(const reco::PFDisplacedVertexRef& ref, Flags type) {

  if( particleId_ != h ) {
    string err;
    err += "PFCandidate::setDisplacedVertexRef: this is not a hadron! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }
  else if(  !flag( T_FROM_DISP ) && !flag( T_TO_DISP ) ) {
    string err;
    err += "PFCandidate::setDisplacedVertexRef: particule flag is neither T_FROM_DISP nor T_TO_DISP";

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }

  if (type == T_TO_DISP && flag( T_TO_DISP )) displacedVertexDaughterRef_ = ref; 
  else if (type == T_FROM_DISP && flag( T_FROM_DISP )) displacedVertexMotherRef_ = ref; 
  else if ( (type == T_FROM_DISP && !flag( T_FROM_DISP )) 
	    || 
	    (type == T_TO_DISP && !flag( T_TO_DISP )) ){
    string err;
    err += "PFCandidate::setDisplacedVertexRef: particule flag is not switched on";

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }

}


void PFCandidate::setV0Ref(const reco::VertexCompositeCandidateRef& ref) {

  v0Ref_ = ref;

}

bool PFCandidate::overlap(const reco::Candidate & other) const {
    CandidatePtr myPtr = sourceCandidatePtr(0);
    if (myPtr.isNull()) return false;
    for (size_t i = 0, n = other.numberOfSourceCandidatePtrs(); i < n; ++i) {
        CandidatePtr otherPtr = other.sourceCandidatePtr(i);
        if ((otherPtr == myPtr) || 
            (sourcePtr_.isNonnull() && otherPtr.isNonnull() && sourcePtr_->overlap(*otherPtr))) {
                return true;
        }
    }
    return false;
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
  out<<" E/pT/eta/phi " 
     <<c.energy()<<"/"
     <<c.pt()<<"/"
     <<c.eta()<<"/"
     <<c.phi();
  if( c.flag( PFCandidate::T_FROM_DISP ) ) out<<", T_FROM_DISP" << endl;
  else if( c.flag( PFCandidate::T_TO_DISP ) ) out<<", T_TO_DISP" << endl;
  else if( c.flag( PFCandidate::T_FROM_GAMMACONV ) ) out<<", T_FROM_GAMMACONV" << endl;
  else if( c.flag( PFCandidate::GAMMA_TO_GAMMACONV ) ) out<<", GAMMA_TO_GAMMACONV" << endl;
  
  out<<", blocks/iele: ";
  
  PFCandidate::ElementsInBlocks eleInBlocks = c.elementsInBlocks();
  for(unsigned i=0; i<eleInBlocks.size(); i++) {
    PFBlockRef blockRef = eleInBlocks[i].first;
    unsigned indexInBlock = eleInBlocks[i].second;
    
    out<<"("<<blockRef.key()<<"|"<<indexInBlock<<"), ";
  }

  out<<" source:"<<c.sourcePtr_.id()<<"/"<<c.sourcePtr_.key();

//   PFBlockRef blockRef = c.block(); 
//   int blockid = blockRef.key(); 
//   const edm::OwnVector< reco::PFBlockElement >& elements = c.elements();
//   out<< "\t# of elements " << elements.size() 
//      <<" from block " << blockid << endl;

//   // print each element in turn
  
//   for(unsigned ie=0; ie<elements.size(); ie++) {
//     out<<"\t"<< elements[ie] <<endl;
//   }
  
  out<<resetiosflags(ios::right|ios::fixed);
  return out;
}
