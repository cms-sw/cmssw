/* Tau discriminant producer plugins.
 *
 * All plugins have a "simple" (i.e. Pt) discriminant name that is used by the
 * MVA framework.
 *
 * The entire plugin name use to build the plugin is
 *
 * "RecoTauDiscrimination"+<simple name>
 *
 * The macros below build these plugins using the simple unary functions
 * defined in RecoTauDiscriminantFunctions and gives them the correct name.
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantFunctions.h"

#include "FWCore/Framework/interface/MakerMacros.h"

// Macros that builds a Function based plugin using the defined naming conventions.
// Build a plugin that produces a single value
#define TAU_DISC_PLUGIN(DiscriminatorFunction) DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory, reco::tau::RecoTauDiscriminantFunctionPlugin<reco::tau::disc::DiscriminatorFunction>, reco::tau::discPluginName(#DiscriminatorFunction))
// Build a plugin that produces multiple values
#define TAU_VEC_DISC_PLUGIN(DiscriminatorFunction) DEFINE_EDM_PLUGIN(RecoTauDiscriminantPluginFactory, reco::tau::RecoTauDiscriminantVectorFunctionPlugin<reco::tau::disc::DiscriminatorFunction>, reco::tau::discPluginName(#DiscriminatorFunction))

TAU_DISC_PLUGIN(JetEta);
TAU_DISC_PLUGIN(AbsJetEta);
TAU_DISC_PLUGIN(JetWidth);
TAU_DISC_PLUGIN(JetTauDR);

// HPStanc variables
TAU_DISC_PLUGIN(JetPt);
TAU_DISC_PLUGIN(SignalPtFraction);
TAU_DISC_PLUGIN(IsolationChargedPtFraction);
TAU_DISC_PLUGIN(IsolationECALPtFraction);
TAU_DISC_PLUGIN(IsolationNeutralHadronPtFraction);
TAU_DISC_PLUGIN(MainTrackPtFraction);
TAU_DISC_PLUGIN(IsolationChargedAveragePtFraction);
TAU_DISC_PLUGIN(ScaledEtaJetCollimation);
TAU_DISC_PLUGIN(ScaledPhiJetCollimation);
TAU_DISC_PLUGIN(ScaledOpeningDeltaR);
TAU_VEC_DISC_PLUGIN(Dalitz2);


TAU_DISC_PLUGIN(IsolationChargedSumHard);
TAU_DISC_PLUGIN(IsolationChargedSumSoft);
TAU_DISC_PLUGIN(IsolationChargedSumHardRelative);
TAU_DISC_PLUGIN(IsolationChargedSumSoftRelative);
TAU_DISC_PLUGIN(IsolationECALSumHard);
TAU_DISC_PLUGIN(IsolationECALSumSoft);
TAU_DISC_PLUGIN(IsolationECALSumHardRelative);
TAU_DISC_PLUGIN(IsolationECALSumSoftRelative);
TAU_DISC_PLUGIN(EMFraction);
TAU_DISC_PLUGIN(ImpactParameterSignificance);

TAU_DISC_PLUGIN(Pt);
TAU_DISC_PLUGIN(Eta);
TAU_DISC_PLUGIN(AbsEta);
TAU_DISC_PLUGIN(Mass);
TAU_DISC_PLUGIN(DecayMode);
TAU_DISC_PLUGIN(OutlierN);
TAU_DISC_PLUGIN(OutlierNCharged);
TAU_DISC_PLUGIN(MainTrackPt);
TAU_DISC_PLUGIN(MainTrackEta);
TAU_DISC_PLUGIN(MainTrackAngle);
TAU_DISC_PLUGIN(OutlierSumPt);
TAU_DISC_PLUGIN(ChargedOutlierSumPt);
TAU_DISC_PLUGIN(NeutralOutlierSumPt);

TAU_DISC_PLUGIN(InvariantMassOfSignal); // obsolete

TAU_VEC_DISC_PLUGIN(TrackPt);
TAU_VEC_DISC_PLUGIN(TrackAngle);
TAU_VEC_DISC_PLUGIN(TrackEta);
TAU_VEC_DISC_PLUGIN(PiZeroPt);
TAU_VEC_DISC_PLUGIN(PiZeroAngle);
TAU_VEC_DISC_PLUGIN(PiZeroEta);
TAU_VEC_DISC_PLUGIN(OutlierPt);
TAU_VEC_DISC_PLUGIN(OutlierAngle);
TAU_VEC_DISC_PLUGIN(ChargedOutlierPt);
TAU_VEC_DISC_PLUGIN(ChargedOutlierAngle);
TAU_VEC_DISC_PLUGIN(NeutralOutlierPt);
TAU_VEC_DISC_PLUGIN(NeutralOutlierAngle);
TAU_VEC_DISC_PLUGIN(Dalitz);

// Obsolete functions
TAU_VEC_DISC_PLUGIN(FilteredObjectPt);
TAU_VEC_DISC_PLUGIN(GammaOccupancy);
TAU_VEC_DISC_PLUGIN(GammaPt);
TAU_VEC_DISC_PLUGIN(InvariantMassOfSignalWithFiltered);
TAU_VEC_DISC_PLUGIN(InvariantMass);
TAU_VEC_DISC_PLUGIN(OutlierMass);

#undef TAU_DISC_PLUGIN
#undef TAU_VEC_DISC_PLUGIN
