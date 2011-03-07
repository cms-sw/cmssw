#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
//#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>
#include <iomanip>

using namespace reco;
using namespace std;



const float PFCandidate::bigMva_ = -999.;

namespace {

  template< int INDEX>
  struct CountBits {
    static const unsigned int value = INDEX%2+CountBits< (INDEX>>1) >::value;
  };

  template<>
  struct CountBits<0> {
    static const unsigned int value = 0;
  };
}


static const unsigned int s_refsBefore[]={CountBits<0>::value,CountBits<1>::value,CountBits<2>::value,
                                          CountBits<3>::value,CountBits<4>::value,CountBits<5>::value,
                                          CountBits<6>::value,CountBits<7>::value,CountBits<8>::value,
					  CountBits<9>::value,
					  CountBits<10>::value,CountBits<11>::value,CountBits<12>::value,
                                          CountBits<13>::value,CountBits<14>::value,CountBits<15>::value,
                                          CountBits<16>::value,CountBits<17>::value,CountBits<18>::value,
					  CountBits<19>::value,
					  CountBits<20>::value,CountBits<21>::value,CountBits<22>::value,
                                          CountBits<23>::value,CountBits<24>::value,CountBits<25>::value,
                                          CountBits<26>::value,CountBits<27>::value,CountBits<28>::value,
					  CountBits<29>::value,
					  CountBits<30>::value,CountBits<31>::value,CountBits<32>::value,
                                          CountBits<33>::value,CountBits<34>::value,CountBits<35>::value,
                                          CountBits<36>::value,CountBits<37>::value,CountBits<38>::value,
					  CountBits<39>::value,
					  CountBits<40>::value,CountBits<41>::value,CountBits<42>::value,
                                          CountBits<43>::value,CountBits<44>::value,CountBits<45>::value,
                                          CountBits<46>::value,CountBits<47>::value,CountBits<48>::value,
					  CountBits<49>::value,
					  CountBits<50>::value,CountBits<51>::value,CountBits<52>::value,
                                          CountBits<53>::value,CountBits<54>::value,CountBits<55>::value,
                                          CountBits<56>::value,CountBits<57>::value,CountBits<58>::value,
					  CountBits<59>::value,
					  CountBits<60>::value,CountBits<61>::value,CountBits<62>::value,
                                          CountBits<63>::value,CountBits<64>::value,CountBits<65>::value,
                                          CountBits<66>::value,CountBits<67>::value,CountBits<68>::value,
					  CountBits<69>::value,
					  CountBits<70>::value,CountBits<71>::value,CountBits<72>::value,
                                          CountBits<73>::value,CountBits<74>::value,CountBits<75>::value,
                                          CountBits<76>::value,CountBits<77>::value,CountBits<78>::value,
					  CountBits<79>::value,
					  CountBits<80>::value,CountBits<81>::value,CountBits<82>::value,
                                          CountBits<83>::value,CountBits<84>::value,CountBits<85>::value,
                                          CountBits<86>::value,CountBits<87>::value,CountBits<88>::value,
					  CountBits<89>::value,
					  CountBits<90>::value,CountBits<1>::value,CountBits<2>::value,
                                          CountBits<93>::value,CountBits<4>::value,CountBits<5>::value,
                                          CountBits<96>::value,CountBits<7>::value,CountBits<8>::value,
					  CountBits<99>::value,
					  CountBits<100>::value,CountBits<101>::value,CountBits<102>::value,
                                          CountBits<103>::value,CountBits<104>::value,CountBits<105>::value,
                                          CountBits<106>::value,CountBits<107>::value,CountBits<108>::value,
					  CountBits<109>::value,
					  CountBits<110>::value,CountBits<111>::value,CountBits<112>::value,
                                          CountBits<113>::value,CountBits<114>::value,CountBits<115>::value,
                                          CountBits<116>::value,CountBits<117>::value,CountBits<118>::value,
					  CountBits<119>::value,
					  CountBits<120>::value,CountBits<121>::value,CountBits<122>::value,
                                          CountBits<123>::value,CountBits<124>::value,CountBits<125>::value,
                                          CountBits<126>::value,CountBits<127>::value
};


enum PFRefBits {
  kRefTrackBit=0x1,
  kRefGsfTrackBit=0x2,
  kRefMuonBit=0x4,
  kRefDisplacedVertexDauBit=0x8,
  kRefDisplacedVertexMotBit=0x10,
  kRefConversionBit=0x20,
  kRefV0Bit=0x40,
  kRefGsfElectronBit=0x80,
  kRefPFElectronExtraBit=0x100,
  kRefPhotonBit=0x200,
  kRefPFPhotonExtraBit=0x400,
  kRefSuperClusterBit=0x800
};
enum PFRefMasks {
  kRefTrackMask=0,
  kRefGsfTrackMask=kRefTrackMask+kRefTrackBit,
  kRefMuonMask=kRefGsfTrackMask+kRefGsfTrackBit,
  kRefDisplacedVertexDauMask=kRefMuonMask+kRefMuonBit,
  kRefDisplacedVertexMotMask=kRefDisplacedVertexDauMask+kRefDisplacedVertexDauBit,
  kRefConversionMask=kRefDisplacedVertexMotMask+kRefDisplacedVertexMotBit,
  kRefV0Mask=kRefConversionMask+kRefConversionBit,
  kRefGsfElectronMask=kRefV0Mask+kRefV0Bit,
  kRefPFElectronExtraMask=kRefGsfElectronMask+kRefGsfElectronBit,
  kRefPhotonMask=kRefPFElectronExtraMask+kRefPFElectronExtraBit,
  kRefPFPhotonExtraMask=kRefPhotonMask+kRefPhotonBit,
  kRefSuperClusterMask=kRefPFPhotonExtraMask+kRefPFPhotonExtraBit
};


#define GETREF( _class_, _mask_,_bit_) \
  edm::ProductID prodID; size_t index, aIndex; \
  typedef edm::Ref<std::vector<_class_> > RefType;	  \
  if(getRefInfo(_mask_, _bit_, prodID, index, aIndex) ) { \
    if (refsCollectionCache_.size()==0 || refsCollectionCache_[aIndex]==0) return RefType(prodID, index, getter_); \
    else { \
      const vector<_class_> *t=reinterpret_cast< const vector<_class_>* >(refsCollectionCache_[aIndex]);\
      return RefType(prodID, &((*t)[aIndex]),index,t);\
    } } \
  return RefType() 



PFCandidate::PFCandidate() : 
  particleId_( X ),
  ecalERatio_(1.),
  hcalERatio_(1.),
  rawEcalEnergy_(0.),
  rawHcalEnergy_(0.),
  ps1Energy_(0.),
  ps2Energy_(0.),
  flags_(0), 
  deltaP_(0.), 
  vertexType_(kCandVertex),
  mva_e_pi_(bigMva_),
  mva_e_mu_(bigMva_),
  mva_pi_mu_(bigMva_),
  mva_nothing_gamma_(bigMva_),
  mva_nothing_nh_(bigMva_),
  mva_gamma_nh_(bigMva_),
  getter_(0),storedRefsBitPattern_(0)
{
  
  //setPdgId( translateTypeToPdgId( particleId_ ) );
  setPdgId(0);
  refsInfo_.reserve(3);
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
  ecalERatio_(1.),
  hcalERatio_(1.),
  rawEcalEnergy_(0),
  rawHcalEnergy_(0),
  ps1Energy_(0.),
  ps2Energy_(0.),
  flags_(0),
  deltaP_(0.),
  vertexType_(kCandVertex),
  mva_e_pi_(bigMva_),
  mva_e_mu_(bigMva_),
  mva_pi_mu_(bigMva_),
  mva_nothing_gamma_(bigMva_),
  mva_nothing_nh_(bigMva_),
  mva_gamma_nh_(bigMva_),
  getter_(0),storedRefsBitPattern_(0)
{
  refsInfo_.reserve(3);
  blocksStorage_.reserve(10);
  elementsStorage_.reserve(10);

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
  //  setPdgId( translateTypeToPdgId( particleId_ ) );
  setPdgId(0);
}



PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


void PFCandidate::addElementInBlock( const reco::PFBlockRef& blockref,
                                     unsigned elementIndex ) {
  //elementsInBlocks_.push_back( make_pair(blockref.key(), elementIndex) );
  if (blocksStorage_.size()==0)
    blocksStorage_ =Blocks(blockref.id());
  blocksStorage_.push_back(blockref);
  elementsStorage_.push_back(elementIndex);
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

static unsigned long long bitPackRefInfo(const edm::RefCore& iCore, size_t iIndex){
  unsigned long long bitPack = iIndex;
  bitPack |= static_cast<unsigned long long>(iCore.id().productIndex())<<32;
  bitPack |= static_cast<unsigned long long>(iCore.id().processIndex())<<48;
  return bitPack;
}

void PFCandidate::storeRefInfo(unsigned int iMask, unsigned int iBit, bool iIsValid, 
			   const edm::RefCore& iCore, size_t iKey, 
			   const edm::EDProductGetter* iGetter) {

  size_t index = s_refsBefore[storedRefsBitPattern_ & iMask];
  if ( 0 == getter_) {
    getter_ = iGetter;
  }

  if(iIsValid) {
    if(0 == (storedRefsBitPattern_ & iBit) ) {
      refsInfo_.insert(refsInfo_.begin()+index, bitPackRefInfo(iCore,iKey));
      if (iGetter==0)
	refsCollectionCache_.insert(refsCollectionCache_.begin()+index,(void*)iCore.productPtr());
      else
	refsCollectionCache_.insert(refsCollectionCache_.begin()+index,0);
    } else {
      assert(refsInfo_.size()>index);
      *(refsInfo_.begin()+index)=bitPackRefInfo(iCore,iKey);
      if (iGetter==0)
	*(refsCollectionCache_.begin()+index)=(void*)iCore.productPtr();
      else
	*(refsCollectionCache_.begin()+index)=0;

    }
    storedRefsBitPattern_ |= iBit;
  } else{
    if( storedRefsBitPattern_ & iBit) {
      refsInfo_.erase(refsInfo_.begin()+index);
      refsCollectionCache_.erase(refsCollectionCache_.begin()+index);
      storedRefsBitPattern_ ^= iBit;
    }
  }

}

bool PFCandidate::getRefInfo(unsigned int iMask, unsigned int iBit, 
			     edm::ProductID& oProductID, size_t& oIndex, size_t& aIndex) const {

  if( 0 == (iBit & storedRefsBitPattern_) ) {
    return false;
  }
  aIndex = s_refsBefore[storedRefsBitPattern_ & iMask];
  unsigned long long bitPacked = refsInfo_[aIndex];
  oIndex = bitPacked & 0xFFFFFFFFULL; //low 32 bits are the index
  unsigned short productIndex = (bitPacked & 0x0000FFFF00000000ULL)>>32;
  unsigned short processIndex = (bitPacked & 0xFFFF000000000000ULL)>>48;
  oProductID = edm::ProductID(processIndex,productIndex);
  return true;
}

void PFCandidate::setTrackRef(const reco::TrackRef& iRef) {
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

  storeRefInfo(kRefTrackMask, kRefTrackBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::TrackRef PFCandidate::trackRef() const { GETREF(reco::Track, kRefTrackMask, kRefTrackBit); }


void PFCandidate::setMuonRef(reco::MuonRef const & iRef) {
  if(  trackRef() != iRef->track() ) {
    string err;
    err += "PFCandidate::setMuonRef: inconsistent track references!";
    
    throw cms::Exception("InconsistentReference",
			 err.c_str() );
  }

  storeRefInfo(kRefMuonMask, kRefMuonBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::MuonRef PFCandidate::muonRef() const { GETREF(reco::Muon, kRefMuonMask, kRefMuonBit); }


//////////////
void PFCandidate::setGsfTrackRef(reco::GsfTrackRef const & iRef) {
  if( particleId_ != e ) {
    string err;
    err += "PFCandidate::setGsfTrackRef: this is not an electron ! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }

  storeRefInfo(kRefGsfTrackMask, kRefGsfTrackBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::GsfTrackRef PFCandidate::gsfTrackRef() const { GETREF(reco::GsfTrack, kRefGsfTrackMask, kRefGsfTrackBit); }


//////////////
void PFCandidate::setDisplacedVertexRef(const reco::PFDisplacedVertexRef& iRef, Flags type) {

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


  if (type == T_TO_DISP && flag( T_TO_DISP )) 
    storeRefInfo(kRefDisplacedVertexDauMask, kRefDisplacedVertexDauBit, 
		 iRef.isNonnull(), 
		 iRef.refCore(), iRef.key(),iRef.productGetter());
  else if (type == T_FROM_DISP && flag( T_FROM_DISP )) 
    storeRefInfo(kRefDisplacedVertexMotMask, kRefDisplacedVertexMotBit, 
		 iRef.isNonnull(), 
		 iRef.refCore(), iRef.key(),iRef.productGetter());
  else if ( (type == T_FROM_DISP && !flag( T_FROM_DISP )) 
	    || 
	    (type == T_TO_DISP && !flag( T_TO_DISP )) ){
    string err;
    err += "PFCandidate::setDisplacedVertexRef: particule flag is not switched on";

    throw cms::Exception("InconsistentReference",
                         err.c_str() );
  }

}




reco::PFDisplacedVertexRef PFCandidate::displacedVertexRef(Flags type) const {
  if (type == T_TO_DISP) {
    GETREF(reco::PFDisplacedVertex, kRefDisplacedVertexDauMask, kRefDisplacedVertexDauBit); 
  }
  else if (type == T_FROM_DISP) {
    GETREF(reco::PFDisplacedVertex, kRefDisplacedVertexMotMask, kRefDisplacedVertexMotBit); 
  }
  return reco::PFDisplacedVertexRef();
}

//////////////
void PFCandidate::setConversionRef(reco::ConversionRef const & iRef) {
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

  storeRefInfo(kRefConversionMask, kRefConversionBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}


reco::ConversionRef PFCandidate::conversionRef() const {
    GETREF(reco::Conversion, kRefConversionMask, kRefConversionBit); 
}

//////////////
void PFCandidate::setV0Ref(reco::VertexCompositeCandidateRef const & iRef) {
  storeRefInfo(kRefV0Mask, kRefV0Bit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::VertexCompositeCandidateRef PFCandidate::v0Ref() const {
  GETREF(reco::VertexCompositeCandidate, kRefV0Mask, kRefV0Bit); 
}

//////////////
void PFCandidate::setGsfElectronRef(reco::GsfElectronRef const & iRef) {
  storeRefInfo(kRefGsfElectronMask, kRefGsfElectronBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::GsfElectronRef PFCandidate::gsfElectronRef() const {
  GETREF(reco::GsfElectron, kRefGsfElectronMask, kRefGsfElectronBit); 
}

//////////////
void PFCandidate::setPFElectronExtraRef(reco::PFCandidateElectronExtraRef const & iRef) {
  storeRefInfo(kRefPFElectronExtraMask, kRefPFElectronExtraBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}

reco::PFCandidateElectronExtraRef PFCandidate::electronExtraRef() const {
  GETREF(reco::PFCandidateElectronExtra, kRefPFElectronExtraMask, kRefPFElectronExtraBit); 
}


reco::PhotonRef PFCandidate::photonRef() const {
  GETREF(reco::Photon, kRefPhotonMask, kRefPhotonBit); 
}

reco::PFCandidatePhotonExtraRef PFCandidate::photonExtraRef() const {
  GETREF(reco::PFCandidatePhotonExtra, kRefPFPhotonExtraMask, kRefPFPhotonExtraBit); 
}

reco::SuperClusterRef PFCandidate::superClusterRef() const {
  GETREF(reco::SuperCluster, kRefSuperClusterMask, kRefSuperClusterBit); 
}

void PFCandidate::setPhotonRef(const reco::PhotonRef& iRef) {
  if( particleId_ != gamma && particleId_ != e) {
    string err;
    err += "PFCandidate::setSuperClusterRef: this is not an electron neither a photon ! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;
    
    throw cms::Exception("InconsistentReference", err.c_str() );
  }

  storeRefInfo(kRefPhotonMask, kRefPhotonBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());

}

void PFCandidate::setSuperClusterRef(const reco::SuperClusterRef& iRef) {
  if( particleId_ != gamma && particleId_ != e) {
    string err;
    err += "PFCandidate::setSuperClusterRef: this is not an electron neither a photon ! particleId_=";
    char num[4];
    sprintf( num, "%d", particleId_);
    err += num;
    
    throw cms::Exception("InconsistentReference", err.c_str() );
  }

  storeRefInfo(kRefSuperClusterMask, kRefSuperClusterBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());

}

void PFCandidate::setPFPhotonExtraRef(const reco::PFCandidatePhotonExtraRef& iRef) {
  storeRefInfo(kRefPFPhotonExtraMask, kRefPFPhotonExtraBit, iRef.isNonnull(), 
	       iRef.refCore(), iRef.key(),iRef.productGetter());
}


    

const math::XYZPoint & PFCandidate::vertex() const {
  switch (vertexType_) {
  case kCandVertex:
    return vertex_;
    break;
  case kTrkVertex:
    return trackRef()->vertex();
    break;
  case kComMuonVertex:
    return muonRef()->combinedMuon()->vertex();
    break;
  case kSAMuonVertex:
    return muonRef()->standAloneMuon()->vertex();
    break;
  case kTrkMuonVertex:
    return muonRef()->track()->vertex();
    break;
  case kGSFVertex:
    return gsfTrackRef()->vertex();
    break;
  }
  return vertex_;
}
