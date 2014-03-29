#ifndef RecoTauTag_RecoTau_RecoTauDiscriminantFunctions_h
#define RecoTauTag_RecoTau_RecoTauDiscriminantFunctions_h

/*
 * RecoTauDiscriminantFunctions
 *
 * Collection of unary functions used to compute tau discriminant values.
 * Each function here (may be) used in an MVA discriminator.
 *
 * The functions all have the form
 *      ReturnType Function(const PFTau& tau)
 * where ReturnType is either vector<double> or double.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "DataFormats/TauReco/interface/PFTau.h"
#include <vector>

namespace reco { namespace tau { namespace disc {

// Save typing
typedef const PFTau& Tau;
typedef std::vector<double> VDouble;

/// For three prong events, take the track that has charge opposite to the
/// composite charge.
PFCandidatePtr mainTrack(const PFTau& tau);

// HPStanc variables
double JetPt(Tau tau);
double JetEta(Tau tau);
double AbsJetEta(Tau tau);
double JetWidth(Tau tau);
// Delta R between tau and jet axis
double JetTauDR(Tau tau);

double SignalPtFraction(Tau tau);
double IsolationChargedPtFraction(Tau tau);
double IsolationECALPtFraction(Tau tau);
double IsolationNeutralHadronPtFraction(Tau tau);
double MainTrackPtFraction(Tau tau);
double IsolationChargedAveragePtFraction(Tau tau);
double OpeningDeltaR(Tau tau);
double OpeningAngle3D(Tau tau);
double ScaledEtaJetCollimation(Tau tau);
double ScaledPhiJetCollimation(Tau tau);
double ScaledOpeningDeltaR(Tau tau);
VDouble Dalitz2(Tau tau);

// Sum of charged isolation activity above/below 1 GeV
double IsolationChargedSumHard(Tau tau);
double IsolationChargedSumSoft(Tau tau);
double IsolationChargedSumHardRelative(Tau tau);
double IsolationChargedSumSoftRelative(Tau tau);

// Sum of ecal isolation activity above/below 1.5 GeV
double IsolationECALSumHard(Tau tau);
double IsolationECALSumSoft(Tau tau);
double IsolationECALSumHardRelative(Tau tau);
double IsolationECALSumSoftRelative(Tau tau);

// Fraction of signal energy carried by pizeros.
double EMFraction(Tau tau);

// Absolute significance of impact parameter
double ImpactParameterSignificance(Tau tau);

double Pt(Tau tau);
double Eta(Tau tau);
double AbsEta(Tau tau);
double Mass(Tau tau);
double DecayMode(Tau tau);

// Number of objects in isolation cone
double OutlierN(Tau);

// Number of charged objects in isolation cone
double OutlierNCharged(Tau);

double OutlierSumPt(Tau);
double ChargedOutlierSumPt(Tau);
double NeutralOutlierSumPt(Tau);

// Pt of the main track
double MainTrackPt(Tau);
// Eta of the main track
double MainTrackEta(Tau);
// Angle of main track to tau axis
double MainTrackAngle(Tau);

// Exactly the same as "Mass", needed for backwards compatability
double InvariantMassOfSignal(Tau tau);

// Quanitites of tracks
VDouble TrackPt(Tau);
VDouble TrackAngle(Tau);
VDouble TrackEta(Tau);

// Quanitites of PiZeros
VDouble PiZeroPt(Tau);
VDouble PiZeroAngle(Tau);
VDouble PiZeroEta(Tau);

// Isolation quantities
VDouble OutlierPt(Tau);
VDouble OutlierAngle(Tau);
VDouble ChargedOutlierPt(Tau);
VDouble ChargedOutlierAngle(Tau);
VDouble NeutralOutlierPt(Tau);
VDouble NeutralOutlierAngle(Tau);

// Dalitz for three prongs
VDouble Dalitz(Tau);

// Deprecated functions needed for backwards compatability
VDouble FilteredObjectPt(Tau);
VDouble GammaOccupancy(Tau);
VDouble GammaPt(Tau);
VDouble InvariantMassOfSignalWithFiltered(Tau);
VDouble InvariantMass(Tau);
VDouble OutlierMass(Tau);

}}} // end namespace reco::tau::disc
#endif
