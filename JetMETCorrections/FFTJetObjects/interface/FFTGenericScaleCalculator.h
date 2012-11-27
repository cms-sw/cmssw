#ifndef JetMETCorrections_FFTJetObjects_FFTGenericScaleCalculator_h
#define JetMETCorrections_FFTJetObjects_FFTGenericScaleCalculator_h

//
// Generic variable mapper for FFTJet jet corrections
//
#include <cmath>
#include <vector>

#include "JetMETCorrections/FFTJetObjects/interface/AbsFFTSpecificScaleCalculator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FFTGenericScaleCalculator : public AbsFFTSpecificScaleCalculator
{
public:
    FFTGenericScaleCalculator(const edm::ParameterSet& ps);

    inline virtual ~FFTGenericScaleCalculator() {}

    virtual void mapFFTJet(const reco::Jet& jet,
                           const reco::FFTJet<float>& fftJet,
                           const math::XYZTLorentzVector& current,
                           double* buf, unsigned dim) const;
private:
    inline double f_safeLog(const double x) const
    {
        if (x > 0.0)
            return log(x);
        else
            return m_minLog;
    }

    std::vector<double> m_factors;
    double m_minLog;

    // Variables from the "current" Lorentz vector
    int m_eta;
    int m_phi;
    int m_pt;
    int m_logPt;
    int m_mass;
    int m_logMass;
    int m_energy;
    int m_logEnergy;
    int m_gamma;
    int m_logGamma;

    // Variables from fftJet
    int m_pileup;
    int m_ncells;
    int m_etSum;
    int m_etaWidth;
    int m_phiWidth;
    int m_averageWidth;
    int m_widthRatio;
    int m_etaPhiCorr;
    int m_fuzziness;
    int m_convergenceDistance;
    int m_recoScale;
    int m_recoScaleRatio;
    int m_membershipFactor;

    // Variables from the precluster
    int m_magnitude;
    int m_logMagnitude;
    int m_magS1;
    int m_LogMagS1;
    int m_magS2;
    int m_LogMagS2;
    int m_driftSpeed;
    int m_magSpeed;
    int m_lifetime;
    int m_scale;
    int m_logScale;
    int m_nearestNeighborDistance;
    int m_clusterRadius;
    int m_clusterSeparation;
    int m_dRFromJet;
    int m_LaplacianS1;
    int m_LaplacianS2;
    int m_LaplacianS3;
    int m_HessianS2;
    int m_HessianS4;
    int m_HessianS6;

    // Variables from reco::Jet
    int m_nConstituents;
    int m_aveConstituentPt;
    int m_logAveConstituentPt;
    int m_constituentPtDistribution;
    int m_constituentEtaPhiSpread;

    // Variables from reco::PFJet
    int m_chargedHadronEnergyFraction;
    int m_neutralHadronEnergyFraction;
    int m_photonEnergyFraction;
    int m_electronEnergyFraction;
    int m_muonEnergyFraction;
    int m_HFHadronEnergyFraction;
    int m_HFEMEnergyFraction;
    int m_chargedHadronMultiplicity;
    int m_neutralHadronMultiplicity;
    int m_photonMultiplicity;
    int m_electronMultiplicity;
    int m_muonMultiplicity;
    int m_HFHadronMultiplicity;
    int m_HFEMMultiplicity;
    int m_chargedEmEnergyFraction;
    int m_chargedMuEnergyFraction;
    int m_neutralEmEnergyFraction;
    int m_EmEnergyFraction;
    int m_chargedMultiplicity;
    int m_neutralMultiplicity;
};

#endif // JetMETCorrections_FFTJetObjects_FFTGenericScaleCalculator_h
