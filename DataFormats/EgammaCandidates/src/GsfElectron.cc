#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

using namespace reco;

GsfElectronCoreRef GsfElectron::core() const { return core_; }

void GsfElectron::init() {
  passCutBasedPreselection_ = false;
  passPflowPreselection_ = false;
  passMvaPreslection_ = false;
  ambiguous_ = true;
  class_ = UNKNOWN;
}

GsfElectron::GsfElectron() { init(); }

GsfElectron::GsfElectron(const GsfElectronCoreRef& core) : core_(core) { init(); }

GsfElectron::GsfElectron(int charge,
                         const ChargeInfo& chargeInfo,
                         const GsfElectronCoreRef& core,
                         const TrackClusterMatching& tcm,
                         const TrackExtrapolations& te,
                         const ClosestCtfTrack& ctfInfo,
                         const FiducialFlags& ff,
                         const ShowerShape& ss,
                         const ConversionRejection& crv)
    : chargeInfo_(chargeInfo),
      core_(core),
      trackClusterMatching_(tcm),
      trackExtrapolations_(te),
      //closestCtfTrack_(ctfInfo),
      fiducialFlags_(ff),
      showerShape_(ss),
      conversionRejection_(crv) {
  init();
  setCharge(charge);
  setVertex(math::XYZPoint(te.positionAtVtx.x(), te.positionAtVtx.y(), te.positionAtVtx.z()));
  setPdgId(-11 * charge);
  /*if (ecalDrivenSeed())*/ corrections_.correctedEcalEnergy = superCluster()->energy();
  //  assert(ctfInfo.ctfTrack==(GsfElectron::core()->ctfTrack())) ;
  //  assert(ctfInfo.shFracInnerHits==(GsfElectron::core()->ctfGsfOverlap())) ;
}

GsfElectron::GsfElectron(int charge,
                         const ChargeInfo& chargeInfo,
                         const GsfElectronCoreRef& core,
                         const TrackClusterMatching& tcm,
                         const TrackExtrapolations& te,
                         const ClosestCtfTrack& ctfInfo,
                         const FiducialFlags& ff,
                         const ShowerShape& ss,
                         const ShowerShape& full5x5_ss,
                         const ConversionRejection& crv,
                         const SaturationInfo& si)
    : chargeInfo_(chargeInfo),
      core_(core),
      trackClusterMatching_(tcm),
      trackExtrapolations_(te),
      fiducialFlags_(ff),
      showerShape_(ss),
      full5x5_showerShape_(full5x5_ss),
      saturationInfo_(si),
      conversionRejection_(crv) {
  init();
  setCharge(charge);
  setVertex(math::XYZPoint(te.positionAtVtx.x(), te.positionAtVtx.y(), te.positionAtVtx.z()));
  setPdgId(-11 * charge);
  corrections_.correctedEcalEnergy = superCluster()->energy();
}

GsfElectron::GsfElectron(const GsfElectron& electron, const GsfElectronCoreRef& core)
    : RecoCandidate(electron),
      chargeInfo_(electron.chargeInfo_),
      core_(core),
      trackClusterMatching_(electron.trackClusterMatching_),
      trackExtrapolations_(electron.trackExtrapolations_),
      //closestCtfTrack_(electron.closestCtfTrack_),
      fiducialFlags_(electron.fiducialFlags_),
      showerShape_(electron.showerShape_),
      full5x5_showerShape_(electron.full5x5_showerShape_),
      saturationInfo_(electron.saturationInfo_),
      dr03_(electron.dr03_),
      dr04_(electron.dr04_),
      conversionRejection_(electron.conversionRejection_),
      pfIso_(electron.pfIso_),
      mvaInput_(electron.mvaInput_),
      mvaOutput_(electron.mvaOutput_),
      passCutBasedPreselection_(electron.passCutBasedPreselection_),
      passPflowPreselection_(electron.passPflowPreselection_),
      passMvaPreslection_(electron.passMvaPreslection_),
      ambiguous_(electron.ambiguous_),
      ambiguousGsfTracks_(electron.ambiguousGsfTracks_),
      classVariables_(electron.classVariables_),
      class_(electron.class_),
      corrections_(electron.corrections_),
      pixelMatchVariables_(electron.pixelMatchVariables_) {
  //assert(electron.core()->ctfTrack()==core->ctfTrack()) ;
  //assert(electron.core()->ctfGsfOverlap()==core->ctfGsfOverlap()) ;
}

GsfElectron::GsfElectron(const GsfElectron& electron,
                         const GsfElectronCoreRef& core,
                         const CaloClusterPtr& electronCluster,
                         const TrackRef& closestCtfTrack,
                         const TrackBaseRef& conversionPartner,
                         const GsfTrackRefVector& ambiguousTracks)
    : RecoCandidate(electron),
      chargeInfo_(electron.chargeInfo_),
      core_(core),
      trackClusterMatching_(electron.trackClusterMatching_),
      trackExtrapolations_(electron.trackExtrapolations_),
      //closestCtfTrack_(electron.closestCtfTrack_),
      fiducialFlags_(electron.fiducialFlags_),
      showerShape_(electron.showerShape_),
      full5x5_showerShape_(electron.full5x5_showerShape_),
      saturationInfo_(electron.saturationInfo_),
      dr03_(electron.dr03_),
      dr04_(electron.dr04_),
      conversionRejection_(electron.conversionRejection_),
      pfIso_(electron.pfIso_),
      mvaInput_(electron.mvaInput_),
      mvaOutput_(electron.mvaOutput_),
      passCutBasedPreselection_(electron.passCutBasedPreselection_),
      passPflowPreselection_(electron.passPflowPreselection_),
      passMvaPreslection_(electron.passMvaPreslection_),
      ambiguous_(electron.ambiguous_),
      ambiguousGsfTracks_(ambiguousTracks),
      //mva_(electron.mva_),
      classVariables_(electron.classVariables_),
      class_(electron.class_),
      corrections_(electron.corrections_),
      pixelMatchVariables_(electron.pixelMatchVariables_) {
  trackClusterMatching_.electronCluster = electronCluster;
  //closestCtfTrack_.ctfTrack = closestCtfTrack ;
  conversionRejection_.partner = conversionPartner;
  //assert(closestCtfTrack==core->ctfTrack()) ;
  //assert(electron.core()->ctfGsfOverlap()==core->ctfGsfOverlap()) ;
  // TO BE DONE
  // Check that the new edm references are really
  // the clones of the former references, and therefore other attributes
  // stay valid :
  // * electron.core_ ~ core ?
  // * electron.trackClusterMatching_.electronCluster ~ electronCluster ?
  // * electron.closestCtfTrack_.ctfTrack ~ closestCtfTrack ?
  // * electron.ambiguousGsfTracks_ ~ ambiguousTracks ?
}

bool GsfElectron::overlap(const Candidate& c) const {
  const RecoCandidate* o = dynamic_cast<const RecoCandidate*>(&c);
  return (o != nullptr && (checkOverlap(gsfTrack(), o->gsfTrack()) || checkOverlap(superCluster(), o->superCluster())));
  //?? return false;
}

GsfElectron* GsfElectron::clone() const { return new GsfElectron(*this); }

GsfElectron* GsfElectron::clone(const GsfElectronCoreRef& core,
                                const CaloClusterPtr& electronCluster,
                                const TrackRef& closestCtfTrack,
                                const TrackBaseRef& conversionPartner,
                                const GsfTrackRefVector& ambiguousTracks) const {
  return new GsfElectron(*this, core, electronCluster, closestCtfTrack, conversionPartner, ambiguousTracks);
}

bool GsfElectron::ecalDriven() const { return (ecalDrivenSeed() && passingCutBasedPreselection()); }

void GsfElectron::setCorrectedEcalEnergyError(float energyError) {
  corrections_.correctedEcalEnergyError = energyError;
}

void GsfElectron::setCorrectedEcalEnergy(float newEnergy) { setCorrectedEcalEnergy(newEnergy, true); }

void GsfElectron::setCorrectedEcalEnergy(float newEnergy, bool rescaleDependentValues) {
  math::XYZTLorentzVectorD momentum = p4();
  momentum *= newEnergy / momentum.e();
  setP4(momentum);
  if (corrections_.correctedEcalEnergy > 0. && rescaleDependentValues) {
    showerShape_.hcalDepth1OverEcal *= corrections_.correctedEcalEnergy / newEnergy;
    showerShape_.hcalDepth2OverEcal *= corrections_.correctedEcalEnergy / newEnergy;
    trackClusterMatching_.eSuperClusterOverP *= newEnergy / corrections_.correctedEcalEnergy;
    corrections_.correctedEcalEnergyError *= newEnergy / corrections_.correctedEcalEnergy;
  }
  corrections_.correctedEcalEnergy = newEnergy;
  corrections_.isEcalEnergyCorrected = true;
}

void GsfElectron::setTrackMomentumError(float trackErr) { corrections_.trackMomentumError = trackErr; }

void GsfElectron::setP4(P4Kind kind, const reco::Candidate::LorentzVector& p4, float error, bool setCandidate) {
  switch (kind) {
    case P4_FROM_SUPER_CLUSTER:
      corrections_.fromSuperClusterP4 = p4;
      corrections_.fromSuperClusterP4Error = error;
      break;
    case P4_COMBINATION:
      corrections_.combinedP4 = p4;
      corrections_.combinedP4Error = error;
      break;
    case P4_PFLOW_COMBINATION:
      corrections_.pflowP4 = p4;
      corrections_.pflowP4Error = error;
      break;
    default:
      throw cms::Exception("GsfElectron") << "unexpected p4 kind: " << kind;
  }
  if (setCandidate) {
    setP4(p4);
    corrections_.candidateP4Kind = kind;
  }
}

const Candidate::LorentzVector& GsfElectron::p4(P4Kind kind) const {
  switch (kind) {
    case P4_FROM_SUPER_CLUSTER:
      return corrections_.fromSuperClusterP4;
    case P4_COMBINATION:
      return corrections_.combinedP4;
    case P4_PFLOW_COMBINATION:
      return corrections_.pflowP4;
    default:
      throw cms::Exception("GsfElectron") << "unexpected p4 kind: " << kind;
  }
}

float GsfElectron::p4Error(P4Kind kind) const {
  switch (kind) {
    case P4_FROM_SUPER_CLUSTER:
      return corrections_.fromSuperClusterP4Error;
    case P4_COMBINATION:
      return corrections_.combinedP4Error;
    case P4_PFLOW_COMBINATION:
      return corrections_.pflowP4Error;
    default:
      throw cms::Exception("GsfElectron") << "unexpected p4 kind: " << kind;
  }
}
