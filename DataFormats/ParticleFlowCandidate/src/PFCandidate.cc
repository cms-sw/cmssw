#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
//#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>
#include <iomanip>

using namespace reco;
using namespace std;

const float PFCandidate::bigMva_ = -999.;

#include "DataFormats/ParticleFlowCandidate/src/CountBits.h"

PFCandidate::PFCandidate()
    : elementsInBlocks_(nullptr),
      ecalERatio_(1.),
      hcalERatio_(1.),
      hoERatio_(1.),
      rawEcalEnergy_(0.),
      rawHcalEnergy_(0.),
      rawHoEnergy_(0.),
      ps1Energy_(0.),
      ps2Energy_(0.),
      flags_(0),
      deltaP_(0.),
      vertexType_(kCandVertex),
      mva_Isolated_(bigMva_),
      mva_e_pi_(bigMva_),
      mva_e_mu_(bigMva_),
      mva_pi_mu_(bigMva_),
      mva_nothing_gamma_(bigMva_),
      mva_nothing_nh_(bigMva_),
      mva_gamma_nh_(bigMva_),
      getter_(nullptr),
      storedRefsBitPattern_(0),
      time_(0.f),
      timeError_(-1.f) {
  muonTrackType_ = reco::Muon::None;

  setPdgId(translateTypeToPdgId(X));
  refsInfo_.reserve(3);
  std::fill(hcalDepthEnergyFractions_.begin(), hcalDepthEnergyFractions_.end(), 0.f);
}

const math::XYZPoint& PFCandidate::vertex() const { return vertexLegacy(vertexType_); }

PFCandidate::PFCandidate(const PFCandidatePtr& sourcePtr) : PFCandidate(*sourcePtr) {
  sourcePtr_ = sourcePtr;
  hcalDepthEnergyFractions_ = sourcePtr->hcalDepthEnergyFractions_;  // GP not sure it's needed
}

PFCandidate::PFCandidate(Charge charge, const LorentzVector& p4, ParticleType partId)
    : CompositeCandidate(charge, p4),
      elementsInBlocks_(nullptr),
      ecalERatio_(1.),
      hcalERatio_(1.),
      hoERatio_(1.),
      rawEcalEnergy_(0.),
      rawHcalEnergy_(0.),
      rawHoEnergy_(0.),
      ps1Energy_(0.),
      ps2Energy_(0.),
      flags_(0),
      deltaP_(0.),
      vertexType_(kCandVertex),
      mva_Isolated_(bigMva_),
      mva_e_pi_(bigMva_),
      mva_e_mu_(bigMva_),
      mva_pi_mu_(bigMva_),
      mva_nothing_gamma_(bigMva_),
      mva_nothing_nh_(bigMva_),
      mva_gamma_nh_(bigMva_),
      getter_(nullptr),
      storedRefsBitPattern_(0),
      time_(0.f),
      timeError_(-1.f) {
  refsInfo_.reserve(3);
  blocksStorage_.reserve(10);
  elementsStorage_.reserve(10);
  std::fill(hcalDepthEnergyFractions_.begin(), hcalDepthEnergyFractions_.end(), 0.f);

  muonTrackType_ = reco::Muon::None;

  // proceed with various consistency checks

  // charged candidate: track ref and charge must be non null
  if (partId == h || partId == e || partId == mu) {
    if (charge == 0) {
      string err;
      err += "Attempt to construct a charged PFCandidate with a zero charge";
      throw cms::Exception("InconsistentValue", err.c_str());
    }
  } else {
    if (charge) {
      string err;
      err += "Attempt to construct a neutral PFCandidate ";
      err += "with a non-zero charge";
      throw cms::Exception("InconsistentValue", err.c_str());
    }
  }
  setPdgId(translateTypeToPdgId(partId));
}

PFCandidate::PFCandidate(PFCandidate const& iOther)
    : CompositeCandidate(iOther),
      elementsInBlocks_(nullptr),
      blocksStorage_(iOther.blocksStorage_),
      elementsStorage_(iOther.elementsStorage_),
      sourcePtr_(iOther.sourcePtr_),
      muonTrackType_(iOther.muonTrackType_),
      ecalERatio_(iOther.ecalERatio_),
      hcalERatio_(iOther.hcalERatio_),
      hoERatio_(iOther.hoERatio_),
      rawEcalEnergy_(iOther.rawEcalEnergy_),
      rawHcalEnergy_(iOther.rawHcalEnergy_),
      rawHoEnergy_(iOther.rawHoEnergy_),
      ps1Energy_(iOther.ps1Energy_),
      ps2Energy_(iOther.ps2Energy_),
      flags_(iOther.flags_),
      deltaP_(iOther.deltaP_),
      vertexType_(iOther.vertexType_),
      mva_Isolated_(iOther.mva_Isolated_),
      mva_e_pi_(iOther.mva_e_pi_),
      mva_e_mu_(iOther.mva_e_mu_),
      mva_pi_mu_(iOther.mva_pi_mu_),
      mva_nothing_gamma_(iOther.mva_nothing_gamma_),
      mva_nothing_nh_(iOther.mva_nothing_nh_),
      mva_gamma_nh_(iOther.mva_gamma_nh_),
      positionAtECALEntrance_(iOther.positionAtECALEntrance_),
      getter_(iOther.getter_),
      storedRefsBitPattern_(iOther.storedRefsBitPattern_),
      refsInfo_(iOther.refsInfo_),
      refsCollectionCache_(iOther.refsCollectionCache_),
      time_(iOther.time_),
      timeError_(iOther.timeError_),
      hcalDepthEnergyFractions_(iOther.hcalDepthEnergyFractions_) {
  auto tmp = iOther.elementsInBlocks_.load(std::memory_order_acquire);
  if (nullptr != tmp) {
    elementsInBlocks_.store(new ElementsInBlocks{*tmp}, std::memory_order_release);
  }
}

PFCandidate& PFCandidate::operator=(PFCandidate const& iOther) {
  CompositeCandidate::operator=(iOther);
  auto tmp = iOther.elementsInBlocks_.load(std::memory_order_acquire);
  if (nullptr != tmp) {
    delete elementsInBlocks_.exchange(new ElementsInBlocks{*tmp}, std::memory_order_acq_rel);
  } else {
    delete elementsInBlocks_.exchange(nullptr, std::memory_order_acq_rel);
  }
  blocksStorage_ = iOther.blocksStorage_;
  elementsStorage_ = iOther.elementsStorage_;
  sourcePtr_ = iOther.sourcePtr_;
  muonTrackType_ = iOther.muonTrackType_;
  ecalERatio_ = iOther.ecalERatio_;
  hcalERatio_ = iOther.hcalERatio_;
  hoERatio_ = iOther.hoERatio_;
  rawEcalEnergy_ = iOther.rawEcalEnergy_;
  rawHcalEnergy_ = iOther.rawHcalEnergy_;
  rawHoEnergy_ = iOther.rawHoEnergy_;
  ps1Energy_ = iOther.ps1Energy_;
  ps2Energy_ = iOther.ps2Energy_;
  flags_ = iOther.flags_;
  deltaP_ = iOther.deltaP_;
  vertexType_ = iOther.vertexType_;
  mva_Isolated_ = iOther.mva_Isolated_;
  mva_e_pi_ = iOther.mva_e_pi_;
  mva_e_mu_ = iOther.mva_e_mu_;
  mva_pi_mu_ = iOther.mva_pi_mu_;
  mva_nothing_gamma_ = iOther.mva_nothing_gamma_;
  mva_nothing_nh_ = iOther.mva_nothing_nh_;
  mva_gamma_nh_ = iOther.mva_gamma_nh_;
  positionAtECALEntrance_ = iOther.positionAtECALEntrance_;
  getter_ = iOther.getter_;
  storedRefsBitPattern_ = iOther.storedRefsBitPattern_;
  refsInfo_ = iOther.refsInfo_;
  refsCollectionCache_ = iOther.refsCollectionCache_;
  time_ = iOther.time_;
  timeError_ = iOther.timeError_;
  hcalDepthEnergyFractions_ = iOther.hcalDepthEnergyFractions_;
  return *this;
}

PFCandidate::~PFCandidate() { delete elementsInBlocks_.load(std::memory_order_acquire); }

PFCandidate* PFCandidate::clone() const { return new PFCandidate(*this); }

void PFCandidate::addElementInBlock(const reco::PFBlockRef& blockref, unsigned elementIndex) {
  //elementsInBlocks_.push_back( make_pair(blockref.key(), elementIndex) );
  if (blocksStorage_.empty())
    blocksStorage_ = Blocks(blockref.id());
  blocksStorage_.push_back(blockref);
  elementsStorage_.push_back(elementIndex);
  auto ptr = elementsInBlocks_.exchange(nullptr);
  delete ptr;
}

PFCandidate::ParticleType PFCandidate::translatePdgIdToType(int pdgid) const {
  switch (std::abs(pdgid)) {
    case 211:
      return h;
    case 11:
      return e;
    case 13:
      return mu;
    case 22:
      return gamma;
    case 130:
      return h0;
    case 1:
      return h_HF;
    case 2:
      return egamma_HF;
    case 0:
      return X;
    default:
      return X;
  }
}

int PFCandidate::translateTypeToPdgId(ParticleType type) const {
  int thecharge = charge();

  switch (type) {
    case h:
      return thecharge * 211;  // pi+
    case e:
      return thecharge * (-11);
    case mu:
      return thecharge * (-13);
    case gamma:
      return 22;
    case h0:
      return 130;  // K_L0
    case h_HF:
      return 1;  // dummy pdg code
    case egamma_HF:
      return 2;  // dummy pdg code
    case X:
    default:
      return 0;
  }
}

void PFCandidate::setParticleType(ParticleType type) { setPdgId(translateTypeToPdgId(type)); }

bool PFCandidate::overlap(const reco::Candidate& other) const {
  CandidatePtr myPtr = sourceCandidatePtr(0);
  if (myPtr.isNull())
    return false;
  for (size_t i = 0, n = other.numberOfSourceCandidatePtrs(); i < n; ++i) {
    CandidatePtr otherPtr = other.sourceCandidatePtr(i);
    if ((otherPtr == myPtr) || (sourcePtr_.isNonnull() && otherPtr.isNonnull() && sourcePtr_->overlap(*otherPtr))) {
      return true;
    }
  }
  return false;
}

void PFCandidate::rescaleMomentum(double rescaleFactor) {
  LorentzVector rescaledp4 = p4();
  rescaledp4 *= rescaleFactor;
  setP4(rescaledp4);
}

void PFCandidate::setFlag(Flags theFlag, bool value) {
  if (value)
    flags_ = flags_ | (1 << theFlag);
  else
    flags_ = flags_ ^ (1 << theFlag);
}

bool PFCandidate::flag(Flags theFlag) const { return (flags_ >> theFlag) & 1; }

ostream& reco::operator<<(ostream& out, const PFCandidate& c) {
  if (!out)
    return out;

  out << "\tPFCandidate type: " << c.particleId();
  out << setiosflags(ios::right);
  out << setiosflags(ios::fixed);
  out << setprecision(3);
  out << " E/pT/eta/phi " << c.energy() << "/" << c.pt() << "/" << c.eta() << "/" << c.phi();
  if (c.flag(PFCandidate::T_FROM_DISP))
    out << ", T_FROM_DISP" << endl;
  else if (c.flag(PFCandidate::T_TO_DISP))
    out << ", T_TO_DISP" << endl;
  else if (c.flag(PFCandidate::T_FROM_GAMMACONV))
    out << ", T_FROM_GAMMACONV" << endl;
  else if (c.flag(PFCandidate::GAMMA_TO_GAMMACONV))
    out << ", GAMMA_TO_GAMMACONV" << endl;

  out << ", blocks/iele: ";

  PFCandidate::ElementsInBlocks eleInBlocks = c.elementsInBlocks();
  for (unsigned i = 0; i < eleInBlocks.size(); i++) {
    PFBlockRef blockRef = eleInBlocks[i].first;
    unsigned indexInBlock = eleInBlocks[i].second;

    out << "(" << blockRef.key() << "|" << indexInBlock << "), ";
  }

  out << " source:" << c.sourcePtr_.id() << "/" << c.sourcePtr_.key();

  //   PFBlockRef blockRef = c.block();
  //   int blockid = blockRef.key();
  //   const edm::OwnVector< reco::PFBlockElement >& elements = c.elements();
  //   out<< "\t# of elements " << elements.size()
  //      <<" from block " << blockid << endl;

  //   // print each element in turn

  //   for(unsigned ie=0; ie<elements.size(); ie++) {
  //     out<<"\t"<< elements[ie] <<endl;
  //   }

  // Improved printout for electrons if PFCandidateElectronExtra is available
  if (c.particleId() == PFCandidate::e && c.electronExtraRef().isNonnull() && c.electronExtraRef().isAvailable()) {
    out << std::endl << *(c.electronExtraRef());
  }
  out << resetiosflags(ios::right | ios::fixed);
  return out;
}

static unsigned long long bitPackRefInfo(const edm::RefCore& iCore, size_t iIndex) {
  unsigned long long bitPack = iIndex;
  bitPack |= static_cast<unsigned long long>(iCore.id().productIndex()) << 32;
  bitPack |= static_cast<unsigned long long>(iCore.id().processIndex()) << 48;
  return bitPack;
}

void PFCandidate::storeRefInfo(unsigned int iMask,
                               unsigned int iBit,
                               bool iIsValid,
                               const edm::RefCore& iCore,
                               size_t iKey,
                               const edm::EDProductGetter* iGetter) {
  size_t index = s_refsBefore[storedRefsBitPattern_ & iMask];
  if (nullptr == getter_) {
    getter_ = iGetter;
  }

  if (iIsValid) {
    if (0 == (storedRefsBitPattern_ & iBit)) {
      refsInfo_.insert(refsInfo_.begin() + index, bitPackRefInfo(iCore, iKey));
      if (iGetter == nullptr)
        refsCollectionCache_.insert(refsCollectionCache_.begin() + index, static_cast<void const*>(iCore.productPtr()));
      else
        refsCollectionCache_.insert(refsCollectionCache_.begin() + index, nullptr);
    } else {
      assert(refsInfo_.size() > index);
      *(refsInfo_.begin() + index) = bitPackRefInfo(iCore, iKey);
      if (iGetter == nullptr)
        *(refsCollectionCache_.begin() + index) = static_cast<void const*>(iCore.productPtr());
      else
        *(refsCollectionCache_.begin() + index) = nullptr;
    }
    storedRefsBitPattern_ |= iBit;
  } else {
    if (storedRefsBitPattern_ & iBit) {
      refsInfo_.erase(refsInfo_.begin() + index);
      refsCollectionCache_.erase(refsCollectionCache_.begin() + index);
      storedRefsBitPattern_ ^= iBit;
    }
  }
}

bool PFCandidate::getRefInfo(
    unsigned int iMask, unsigned int iBit, edm::ProductID& oProductID, size_t& oIndex, size_t& aIndex) const {
  if (0 == (iBit & storedRefsBitPattern_)) {
    return false;
  }
  aIndex = s_refsBefore[storedRefsBitPattern_ & iMask];
  unsigned long long bitPacked = refsInfo_[aIndex];
  oIndex = bitPacked & 0xFFFFFFFFULL;  //low 32 bits are the index
  unsigned short productIndex = (bitPacked & 0x0000FFFF00000000ULL) >> 32;
  unsigned short processIndex = (bitPacked & 0xFFFF000000000000ULL) >> 48;
  oProductID = edm::ProductID(processIndex, productIndex);
  return true;
}

void PFCandidate::setTrackRef(const reco::TrackRef& iRef) {
  if (!charge()) {
    string err;
    err += "PFCandidate::setTrackRef: this is a neutral candidate! ";
    err += "particleId_=";
    char num[4];
    sprintf(num, "%d", particleId());
    err += num;

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  storeRefInfo(kRefTrackMask, kRefTrackBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::TrackRef PFCandidate::trackRef() const { GETREF(reco::Track, kRefTrackMask, kRefTrackBit); }

void PFCandidate::setMuonRef(reco::MuonRef const& iRef) {
  if (trackRef() != iRef->track()) {
    string err;
    err += "PFCandidate::setMuonRef: inconsistent track references!";

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  storeRefInfo(kRefMuonMask, kRefMuonBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::MuonRef PFCandidate::muonRef() const { GETREF(reco::Muon, kRefMuonMask, kRefMuonBit); }

//////////////
void PFCandidate::setGsfTrackRef(reco::GsfTrackRef const& iRef) {
  //  Removed by F. Beaudette. Would like to be able to save the GsfTrackRef even for charged pions
  //  if( particleId() != e ) {
  //    string err;
  //    err += "PFCandidate::setGsfTrackRef: this is not an electron ! particleId_=";
  //    char num[4];
  //    sprintf( num, "%d", particleId());
  //    err += num;
  //
  //    throw cms::Exception("InconsistentReference",
  //                         err.c_str() );
  //  }

  storeRefInfo(kRefGsfTrackMask, kRefGsfTrackBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::GsfTrackRef PFCandidate::gsfTrackRef() const { GETREF(reco::GsfTrack, kRefGsfTrackMask, kRefGsfTrackBit); }

//////////////
void PFCandidate::setDisplacedVertexRef(const reco::PFDisplacedVertexRef& iRef, Flags type) {
  if (particleId() != h) {
    string err;
    err += "PFCandidate::setDisplacedVertexRef: this is not a hadron! particleId_=";
    char num[4];
    sprintf(num, "%d", particleId());
    err += num;

    throw cms::Exception("InconsistentReference", err.c_str());
  } else if (!flag(T_FROM_DISP) && !flag(T_TO_DISP)) {
    string err;
    err += "PFCandidate::setDisplacedVertexRef: particule flag is neither T_FROM_DISP nor T_TO_DISP";

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  if (type == T_TO_DISP && flag(T_TO_DISP))
    storeRefInfo(kRefDisplacedVertexDauMask,
                 kRefDisplacedVertexDauBit,
                 iRef.isNonnull(),
                 iRef.refCore(),
                 iRef.key(),
                 iRef.productGetter());
  else if (type == T_FROM_DISP && flag(T_FROM_DISP))
    storeRefInfo(kRefDisplacedVertexMotMask,
                 kRefDisplacedVertexMotBit,
                 iRef.isNonnull(),
                 iRef.refCore(),
                 iRef.key(),
                 iRef.productGetter());
  else if ((type == T_FROM_DISP && !flag(T_FROM_DISP)) || (type == T_TO_DISP && !flag(T_TO_DISP))) {
    string err;
    err += "PFCandidate::setDisplacedVertexRef: particule flag is not switched on";

    throw cms::Exception("InconsistentReference", err.c_str());
  }
}

reco::PFDisplacedVertexRef PFCandidate::displacedVertexRef(Flags type) const {
  if (type == T_TO_DISP) {
    GETREF(reco::PFDisplacedVertex, kRefDisplacedVertexDauMask, kRefDisplacedVertexDauBit);
  } else if (type == T_FROM_DISP) {
    GETREF(reco::PFDisplacedVertex, kRefDisplacedVertexMotMask, kRefDisplacedVertexMotBit);
  }
  return reco::PFDisplacedVertexRef();
}

//////////////
void PFCandidate::setConversionRef(reco::ConversionRef const& iRef) {
  if (particleId() != gamma) {
    string err;
    err += "PFCandidate::setConversionRef: this is not a (converted) photon ! particleId_=";
    char num[4];
    sprintf(num, "%d", particleId());
    err += num;

    throw cms::Exception("InconsistentReference", err.c_str());
  } else if (!flag(GAMMA_TO_GAMMACONV)) {
    string err;
    err += "PFCandidate::setConversionRef: particule flag is not GAMMA_TO_GAMMACONV";

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  storeRefInfo(
      kRefConversionMask, kRefConversionBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::ConversionRef PFCandidate::conversionRef() const {
  GETREF(reco::Conversion, kRefConversionMask, kRefConversionBit);
}

//////////////
void PFCandidate::setV0Ref(reco::VertexCompositeCandidateRef const& iRef) {
  storeRefInfo(kRefV0Mask, kRefV0Bit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::VertexCompositeCandidateRef PFCandidate::v0Ref() const {
  GETREF(reco::VertexCompositeCandidate, kRefV0Mask, kRefV0Bit);
}

//////////////
void PFCandidate::setGsfElectronRef(reco::GsfElectronRef const& iRef) {
  storeRefInfo(
      kRefGsfElectronMask, kRefGsfElectronBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

reco::GsfElectronRef PFCandidate::gsfElectronRef() const {
  GETREF(reco::GsfElectron, kRefGsfElectronMask, kRefGsfElectronBit);
}

//////////////
void PFCandidate::setPFElectronExtraRef(reco::PFCandidateElectronExtraRef const& iRef) {
  storeRefInfo(kRefPFElectronExtraMask,
               kRefPFElectronExtraBit,
               iRef.isNonnull(),
               iRef.refCore(),
               iRef.key(),
               iRef.productGetter());
}

reco::PFCandidateElectronExtraRef PFCandidate::electronExtraRef() const {
  GETREF(reco::PFCandidateElectronExtra, kRefPFElectronExtraMask, kRefPFElectronExtraBit);
}

reco::PhotonRef PFCandidate::photonRef() const { GETREF(reco::Photon, kRefPhotonMask, kRefPhotonBit); }

reco::PFCandidatePhotonExtraRef PFCandidate::photonExtraRef() const {
  GETREF(reco::PFCandidatePhotonExtra, kRefPFPhotonExtraMask, kRefPFPhotonExtraBit);
}

reco::PFCandidateEGammaExtraRef PFCandidate::egammaExtraRef() const {
  GETREF(reco::PFCandidateEGammaExtra, kRefPFEGammaExtraMask, kRefPFEGammaExtraBit);
}

reco::SuperClusterRef PFCandidate::superClusterRef() const {
  GETREF(reco::SuperCluster, kRefSuperClusterMask, kRefSuperClusterBit);
}

void PFCandidate::setPhotonRef(const reco::PhotonRef& iRef) {
  if (particleId() != gamma && particleId() != e) {
    string err;
    err += "PFCandidate::setSuperClusterRef: this is not an electron neither a photon ! particleId_=";
    char num[4];
    sprintf(num, "%d", particleId());
    err += num;

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  storeRefInfo(kRefPhotonMask, kRefPhotonBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

void PFCandidate::setSuperClusterRef(const reco::SuperClusterRef& iRef) {
  if (particleId() != gamma && particleId() != e) {
    string err;
    err += "PFCandidate::setSuperClusterRef: this is not an electron neither a photon ! particleId_=";
    char num[4];
    sprintf(num, "%d", particleId());
    err += num;

    throw cms::Exception("InconsistentReference", err.c_str());
  }

  storeRefInfo(
      kRefSuperClusterMask, kRefSuperClusterBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

void PFCandidate::setPFPhotonExtraRef(const reco::PFCandidatePhotonExtraRef& iRef) {
  storeRefInfo(
      kRefPFPhotonExtraMask, kRefPFPhotonExtraBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

void PFCandidate::setPFEGammaExtraRef(const reco::PFCandidateEGammaExtraRef& iRef) {
  storeRefInfo(
      kRefPFEGammaExtraMask, kRefPFEGammaExtraBit, iRef.isNonnull(), iRef.refCore(), iRef.key(), iRef.productGetter());
}

const math::XYZPoint& PFCandidate::vertexLegacy(PFCandidate::PFVertexType vertexType) const {
  switch (vertexType) {
    case kCandVertex:
      return LeafCandidate::vertex();
      break;
    //the following cases will only be called for legacy AOD which does not have an embedded vertex
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
    case kTPFMSMuonVertex:
      return muonRef()->tpfmsTrack()->vertex();
      break;
    case kPickyMuonVertex:
      return muonRef()->pickyTrack()->vertex();
      break;
    case kDYTMuonVertex:
      return muonRef()->dytTrack()->vertex();
      break;

    case kGSFVertex:
      return gsfTrackRef()->vertex();
      break;
  }
  return LeafCandidate::vertex();
}

const PFCandidate::ElementsInBlocks& PFCandidate::elementsInBlocks() const {
  if (nullptr == elementsInBlocks_.load(std::memory_order_acquire)) {
    std::unique_ptr<ElementsInBlocks> temp(new ElementsInBlocks(blocksStorage_.size()));
    for (unsigned int icopy = 0; icopy != blocksStorage_.size(); ++icopy)
      (*temp)[icopy] = std::make_pair(blocksStorage_[icopy], elementsStorage_[icopy]);
    ElementsInBlocks* expected = nullptr;
    if (elementsInBlocks_.compare_exchange_strong(expected, temp.get(), std::memory_order_acq_rel)) {
      temp.release();
    }
  }
  return *(elementsInBlocks_.load(std::memory_order_acquire));
}
