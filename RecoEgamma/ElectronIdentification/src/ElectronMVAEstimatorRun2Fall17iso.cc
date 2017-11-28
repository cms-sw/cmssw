#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17iso.h"

// A function that should work on both pat and reco objects
std::vector<float> ElectronMVAEstimatorRun2Fall17iso::
fillMVAVariables(const reco::GsfElectron* eleRecoPtr, const edm::Handle<reco::ConversionCollection> conversions, const reco::BeamSpot *theBeamSpot, const edm::Handle<double> rho) const {


  // Both pat and reco particles have exactly the same accessors, so we use a reco ptr
  // throughout the code, with a single exception as of this writing, handled separately below.
  auto superCluster = eleRecoPtr->superCluster();

  // Pure ECAL -> shower shapes
  float see            = eleRecoPtr->full5x5_sigmaIetaIeta();
  float spp            = eleRecoPtr->full5x5_sigmaIphiIphi();
  float OneMinusE1x5E5x5 = 1. - eleRecoPtr->full5x5_e1x5() / eleRecoPtr->full5x5_e5x5();
  float R9             = eleRecoPtr->full5x5_r9();
  float etawidth       = superCluster->etaWidth();
  float phiwidth       = superCluster->phiWidth();
  float HoE            = eleRecoPtr->full5x5_hcalOverEcal(); //hadronicOverEm();
  // Endcap only variables
  float PreShowerOverRaw  = superCluster->preshowerEnergy() / superCluster->rawEnergy();

  // To get to CTF track information in pat::Electron, we have to have the pointer
  // to pat::Electron, it is not accessible from the pointer to reco::GsfElectron.
  // This behavior is reported and is expected to change in the future (post-7.4.5 some time).
  bool validKF= false;
  reco::TrackRef myTrackRef = eleRecoPtr->closestCtfTrackRef();
  const pat::Electron * elePatPtr = dynamic_cast<const pat::Electron *>(eleRecoPtr);
  // Check if this is really a pat::Electron, and if yes, get the track ref from this new
  // pointer instead
  if( elePatPtr != nullptr )
    myTrackRef = elePatPtr->closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable() && (myTrackRef.isNonnull()) );

  //Pure tracking variables
  float kfhits         = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  float kfchi2          = (validKF) ? myTrackRef->normalizedChi2() : 0;
  float gsfchi2         = eleRecoPtr->gsfTrack()->normalizedChi2();

  // Energy matching
  float fbrem           = eleRecoPtr->fbrem();

  float gsfhits         = eleRecoPtr->gsfTrack()->hitPattern().trackerLayersWithMeasurement();
  float expectedMissingInnerHits = eleRecoPtr->gsfTrack()
    ->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

  reco::ConversionRef conv_ref = ConversionTools::matchedConversion(*eleRecoPtr,
                                    conversions,
                                    theBeamSpot->position());
  float convVtxFitProbability = -1.;
  if(!conv_ref.isNull()) {
    const reco::Vertex &vtx = conv_ref.get()->conversionVertex(); if (vtx.isValid()) {
      convVtxFitProbability = (float)TMath::Prob( vtx.chi2(), vtx.ndof());
    }
  }

  float EoP             = eleRecoPtr->eSuperClusterOverP();
  float eleEoPout       = eleRecoPtr->eEleClusterOverPout();
  float pAtVertex            = eleRecoPtr->trackMomentumAtVtx().R();
  float IoEmIoP         = (1.0/eleRecoPtr->ecalEnergy()) - (1.0 / pAtVertex );

  // Isolation variables
  float ele_pfChargedHadIso   = (eleRecoPtr->pfIsolationVariables()).sumChargedHadronPt ; //chargedHadronIso();
  float ele_pfNeutralHadIso   = (eleRecoPtr->pfIsolationVariables()).sumNeutralHadronEt ; //neutralHadronIso();
  float ele_pfPhotonIso       = (eleRecoPtr->pfIsolationVariables()).sumPhotonEt; //photonIso();

  // Geometrical matchings
  float deta            = eleRecoPtr->deltaEtaSuperClusterTrackAtVtx();
  float dphi            = eleRecoPtr->deltaPhiSuperClusterTrackAtVtx();
  float detacalo        = eleRecoPtr->deltaEtaSeedClusterTrackAtCalo();

  std::vector<float> vars = std::move( packMVAVariables(
                                           see,                      // 0
                                           spp,                      // 1
                                           OneMinusE1x5E5x5,
                                           R9,
                                           etawidth,
                                           phiwidth,                 // 5
                                           HoE,
                                           //Pure tracking variables
                                           kfhits,
                                           kfchi2,
                                           gsfchi2,                  // 9
                                           // Energy matching
                                           fbrem,
                                           gsfhits,
                                           expectedMissingInnerHits,
                                           convVtxFitProbability,     // 13
                                           EoP,
                                           eleEoPout,                // 15
                                           IoEmIoP,
                                           // Geometrical matchings
                                           deta,                     // 17
                                           dphi,
                                           detacalo,
                                           // Isolation variables
                                           ele_pfPhotonIso,
                                           ele_pfChargedHadIso,
                                           ele_pfNeutralHadIso,
                                           // Pileup
                                           (float)*rho,
                                           // Endcap only variables
                                           PreShowerOverRaw          // 24
                                      )
                      );

  constrainMVAVariables(vars);

  return vars;
}
