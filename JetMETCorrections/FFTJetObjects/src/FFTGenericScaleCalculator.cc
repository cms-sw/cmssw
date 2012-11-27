#include <cassert>
#include <cfloat>

#include "JetMETCorrections/FFTJetObjects/interface/FFTGenericScaleCalculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#define int_param(varname) m_ ## varname (ps.getParameter< int >( #varname ))

#define check_param(varname) if ((m_ ## varname) >= 0) {\
        if ((m_ ## varname) >= nFactors)\
            throw cms::Exception("FFTJetBadConfig")\
                << "In FFTGenericScaleCalculator constructor: "\
                << "out of range mapping for variable \""\
                <<  #varname << "\"" << std::endl;\
        mask[(m_ ## varname)] = 1;\
        ++dim;\
    }


static inline double delPhi(const double phi1, const double phi2)
{
    double dphi = phi1 - phi2;
    if (dphi > M_PI)
        dphi -= 2.0*M_PI;
    else if (dphi < -M_PI)
        dphi += 2.0*M_PI;
    return dphi;
}


FFTGenericScaleCalculator::FFTGenericScaleCalculator(const edm::ParameterSet& ps)
    : m_factors(ps.getParameter<std::vector<double> >("factors")),
      m_minLog(ps.getUntrackedParameter<double>("minLog", -800.0)),
      int_param(eta),
      int_param(phi),
      int_param(pt),
      int_param(logPt),
      int_param(mass),
      int_param(logMass),
      int_param(energy),
      int_param(logEnergy),
      int_param(gamma),
      int_param(logGamma),
      int_param(pileup),
      int_param(ncells),
      int_param(etSum),
      int_param(etaWidth),
      int_param(phiWidth),
      int_param(averageWidth),
      int_param(widthRatio),
      int_param(etaPhiCorr),
      int_param(fuzziness),
      int_param(convergenceDistance),
      int_param(recoScale),
      int_param(recoScaleRatio),
      int_param(membershipFactor),
      int_param(magnitude),
      int_param(logMagnitude),
      int_param(magS1),
      int_param(LogMagS1),
      int_param(magS2),
      int_param(LogMagS2),
      int_param(driftSpeed),
      int_param(magSpeed),
      int_param(lifetime),
      int_param(scale),
      int_param(logScale),
      int_param(nearestNeighborDistance),
      int_param(clusterRadius),
      int_param(clusterSeparation),
      int_param(dRFromJet),
      int_param(LaplacianS1),
      int_param(LaplacianS2),
      int_param(LaplacianS3),
      int_param(HessianS2),
      int_param(HessianS4),
      int_param(HessianS6),
      int_param(nConstituents),
      int_param(aveConstituentPt),
      int_param(logAveConstituentPt),
      int_param(constituentPtDistribution),
      int_param(constituentEtaPhiSpread),
      int_param(chargedHadronEnergyFraction),
      int_param(neutralHadronEnergyFraction),
      int_param(photonEnergyFraction),
      int_param(electronEnergyFraction),
      int_param(muonEnergyFraction),
      int_param(HFHadronEnergyFraction),
      int_param(HFEMEnergyFraction),
      int_param(chargedHadronMultiplicity),
      int_param(neutralHadronMultiplicity),
      int_param(photonMultiplicity),
      int_param(electronMultiplicity),
      int_param(muonMultiplicity),
      int_param(HFHadronMultiplicity),
      int_param(HFEMMultiplicity),
      int_param(chargedEmEnergyFraction),
      int_param(chargedMuEnergyFraction),
      int_param(neutralEmEnergyFraction),
      int_param(EmEnergyFraction),
      int_param(chargedMultiplicity),
      int_param(neutralMultiplicity)
{
    const int nFactors = m_factors.size();
    std::vector<int> mask(nFactors, 0);
    int dim = 0;

    check_param(eta);
    check_param(phi);
    check_param(pt);
    check_param(logPt);
    check_param(mass);
    check_param(logMass);
    check_param(energy);
    check_param(logEnergy);
    check_param(gamma);
    check_param(logGamma);
    check_param(pileup);
    check_param(ncells);
    check_param(etSum);
    check_param(etaWidth);
    check_param(phiWidth);
    check_param(averageWidth);
    check_param(widthRatio);
    check_param(etaPhiCorr);
    check_param(fuzziness);
    check_param(convergenceDistance);
    check_param(recoScale);
    check_param(recoScaleRatio);
    check_param(membershipFactor);
    check_param(magnitude);
    check_param(logMagnitude);
    check_param(magS1);
    check_param(LogMagS1);
    check_param(magS2);
    check_param(LogMagS2);
    check_param(driftSpeed);
    check_param(magSpeed);
    check_param(lifetime);
    check_param(scale);
    check_param(logScale);
    check_param(nearestNeighborDistance);
    check_param(clusterRadius);
    check_param(clusterSeparation);
    check_param(dRFromJet);
    check_param(LaplacianS1);
    check_param(LaplacianS2);
    check_param(LaplacianS3);
    check_param(HessianS2);
    check_param(HessianS4);
    check_param(HessianS6);
    check_param(nConstituents);
    check_param(aveConstituentPt);
    check_param(logAveConstituentPt);
    check_param(constituentPtDistribution);
    check_param(constituentEtaPhiSpread);
    check_param(chargedHadronEnergyFraction);
    check_param(neutralHadronEnergyFraction);
    check_param(photonEnergyFraction);
    check_param(electronEnergyFraction);
    check_param(muonEnergyFraction);
    check_param(HFHadronEnergyFraction);
    check_param(HFEMEnergyFraction);
    check_param(chargedHadronMultiplicity);
    check_param(neutralHadronMultiplicity);
    check_param(photonMultiplicity);
    check_param(electronMultiplicity);
    check_param(muonMultiplicity);
    check_param(HFHadronMultiplicity);
    check_param(HFEMMultiplicity);
    check_param(chargedEmEnergyFraction);
    check_param(chargedMuEnergyFraction);
    check_param(neutralEmEnergyFraction);
    check_param(EmEnergyFraction);
    check_param(chargedMultiplicity);
    check_param(neutralMultiplicity);

    if (dim != nFactors)
        throw cms::Exception("FFTJetBadConfig")
            << "In FFTGenericScaleCalculator constructor: "
            << "incompatible number of scaling factors: expected "
            << dim << ", got " << nFactors << std::endl;
    for (int i=0; i<nFactors; ++i)
        if (mask[i] == 0)
            throw cms::Exception("FFTJetBadConfig")
                << "In FFTGenericScaleCalculator constructor: "
                << "variable number " << i << " is not mapped" << std::endl;
}

void FFTGenericScaleCalculator::mapFFTJet(
    const reco::Jet& jet, const reco::FFTJet<float>& fftJet,
    const math::XYZTLorentzVector& current,
    double* buf, const unsigned dim) const
{
    // Verify that the input is reasonable
    if (dim != m_factors.size())
        throw cms::Exception("FFTJetBadConfig")
            << "In FFTGenericScaleCalculator::mapFFTJet: "
            << "incompatible table dimensionality: expected "
            << m_factors.size() << ", got " << dim << std::endl;
    if (dim)
    {
        assert(buf);
        for (unsigned i=0; i<dim; ++i)
            buf[i] = 0.0;
    }
    else
        return;

    // Go over all variables and map them as configured.
    // Variables from the "current" Lorentz vector.
    if (m_eta >= 0)
        buf[m_eta] = current.eta();

    if (m_phi >= 0)
        buf[m_phi] = current.phi();

    if (m_pt >= 0)
        buf[m_pt] = current.pt();

    if (m_logPt >= 0)
        buf[m_logPt] = f_safeLog(current.pt());

    if (m_mass >= 0)
        buf[m_mass] = current.M();

    if (m_logMass >= 0)
        buf[m_mass] = f_safeLog(current.M());

    if (m_energy >= 0)
        buf[m_energy] = current.e();

    if (m_logEnergy >= 0)
        buf[m_energy] = f_safeLog(current.e());

    if (m_gamma >= 0)
    {
        const double m = current.M();
        if (m > 0.0)
            buf[m_gamma] = current.e()/m;
        else
            buf[m_gamma] = DBL_MAX;
    }

    if (m_logGamma >= 0)
    {
        const double m = current.M();
        if (m > 0.0)
            buf[m_gamma] = current.e()/m;
        else
            buf[m_gamma] = DBL_MAX;
        buf[m_gamma] = log(buf[m_gamma]);
    }

    // Variables from fftJet
    if (m_pileup >= 0)
        buf[m_pileup] = fftJet.f_pileup().pt();

    if (m_ncells >= 0)
        buf[m_ncells] = fftJet.f_ncells();

    if (m_etSum >= 0)
        buf[m_etSum] = fftJet.f_etSum();

    if (m_etaWidth >= 0)
        buf[m_etaWidth] = fftJet.f_etaWidth();

    if (m_phiWidth >= 0)
        buf[m_phiWidth] = fftJet.f_phiWidth();

    if (m_averageWidth >= 0)
    {
        const double etaw = fftJet.f_etaWidth();
        const double phiw = fftJet.f_phiWidth();
        buf[m_averageWidth] = sqrt(etaw*etaw + phiw*phiw);
    }

    if (m_widthRatio >= 0)
    {
        const double etaw = fftJet.f_etaWidth();
        const double phiw = fftJet.f_phiWidth();
        if (phiw > 0.0)
            buf[m_widthRatio] = etaw/phiw;
        else
            buf[m_widthRatio] = DBL_MAX;
    }

    if (m_etaPhiCorr >= 0)
        buf[m_etaPhiCorr] = fftJet.f_etaPhiCorr();

    if (m_fuzziness >= 0)
        buf[m_fuzziness] = fftJet.f_fuzziness();

    if (m_convergenceDistance >= 0)
        buf[m_convergenceDistance] = fftJet.f_convergenceDistance();

    if (m_recoScale >= 0)
        buf[m_recoScale] = fftJet.f_recoScale();

    if (m_recoScaleRatio >= 0)
        buf[m_recoScaleRatio] = fftJet.f_recoScaleRatio();

    if (m_membershipFactor >= 0)
        buf[m_membershipFactor] = fftJet.f_membershipFactor();

    // Get most often used precluster quantities
    const reco::PattRecoPeak<float>& preclus = fftJet.f_precluster();
    const double scale = preclus.scale();

    if (m_magnitude >= 0)
        buf[m_magnitude] = preclus.magnitude();

    if (m_logMagnitude >= 0)
        buf[m_logMagnitude] = f_safeLog(preclus.magnitude());

    if (m_magS1 >= 0)
        buf[m_magS1] = preclus.magnitude()*scale;

    if (m_LogMagS1 >= 0)
        buf[m_LogMagS1] = f_safeLog(preclus.magnitude()*scale);

    if (m_magS2 >= 0)
        buf[m_magS2] = preclus.magnitude()*scale*scale;

    if (m_LogMagS2 >= 0)
        buf[m_LogMagS2] = f_safeLog(preclus.magnitude()*scale*scale);

    if (m_driftSpeed >= 0)
        buf[m_driftSpeed] = preclus.driftSpeed();

    if (m_magSpeed >= 0)
        buf[m_magSpeed] = preclus.magSpeed();

    if (m_lifetime >= 0)
        buf[m_lifetime] = preclus.lifetime();

    if (m_scale >= 0)
        buf[m_scale] = scale;

    if (m_logScale >= 0)
        buf[m_logScale] = f_safeLog(scale);

    if (m_nearestNeighborDistance >= 0)
        buf[m_nearestNeighborDistance] = preclus.nearestNeighborDistance();

    if (m_clusterRadius >= 0)
        buf[m_clusterRadius] = preclus.clusterRadius();

    if (m_clusterSeparation >= 0)
        buf[m_clusterSeparation] = preclus.clusterSeparation();

    if (m_dRFromJet >= 0)
    {
        const double deta = preclus.eta() - current.eta();
        const double dphi = delPhi(preclus.phi(), current.phi());
        buf[m_dRFromJet] = sqrt(deta*deta + dphi*dphi);
    }

    if (m_LaplacianS1 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_LaplacianS1] = fabs(h[0] + h[2])*scale;
    }

    if (m_LaplacianS2 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_LaplacianS2] = fabs(h[0] + h[2])*scale*scale;
    }

    if (m_LaplacianS3 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_LaplacianS3] = fabs(h[0] + h[2])*scale*scale*scale;
    }

    if (m_HessianS2 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_HessianS2] = fabs(h[0]*h[2] - h[1]*h[1])*scale*scale;
    }

    if (m_HessianS4 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_HessianS4] = fabs(h[0]*h[2] - h[1]*h[1])*pow(scale, 4);
    }

    if (m_HessianS6 >= 0)
    {
        double h[3];
        preclus.hessian(h);
        buf[m_HessianS6] = fabs(h[0]*h[2] - h[1]*h[1])*pow(scale, 6);
    }

    // Variables from reco::Jet
    if (m_nConstituents >= 0)
        buf[m_nConstituents] = jet.nConstituents();

    if (m_aveConstituentPt >= 0)
        buf[m_aveConstituentPt] = current.pt()/jet.nConstituents();

    if (m_logAveConstituentPt >= 0)
        buf[m_logAveConstituentPt] = f_safeLog(current.pt()/jet.nConstituents());

    if (m_constituentPtDistribution >= 0)
        buf[m_constituentPtDistribution] = jet.constituentPtDistribution();

    if (m_constituentEtaPhiSpread >= 0)
        buf[m_constituentEtaPhiSpread] = jet.constituentEtaPhiSpread();

    // Variables from reco::PFJet
    const reco::PFJet* pfjet = dynamic_cast<const reco::PFJet*>(&jet);
    if (pfjet)
    {
        // Particle flow jet
        if (m_chargedHadronEnergyFraction >= 0)
            buf[m_chargedHadronEnergyFraction] = pfjet->chargedHadronEnergyFraction();

        if (m_neutralHadronEnergyFraction >= 0)
            buf[m_neutralHadronEnergyFraction] = pfjet->neutralHadronEnergyFraction();

        if (m_photonEnergyFraction >= 0)
            buf[m_photonEnergyFraction] = pfjet->photonEnergyFraction();

        if (m_electronEnergyFraction >= 0)
            buf[m_electronEnergyFraction] = pfjet->electronEnergyFraction();

        if (m_muonEnergyFraction >= 0)
            buf[m_muonEnergyFraction] = pfjet->muonEnergyFraction();

        if (m_HFHadronEnergyFraction >= 0)
            buf[m_HFHadronEnergyFraction] = pfjet->HFHadronEnergyFraction();

        if (m_HFEMEnergyFraction >= 0)
            buf[m_HFEMEnergyFraction] = pfjet->HFEMEnergyFraction();

        if (m_chargedHadronMultiplicity >= 0)
            buf[m_chargedHadronMultiplicity] = pfjet->chargedHadronMultiplicity();

        if (m_neutralHadronMultiplicity >= 0)
            buf[m_neutralHadronMultiplicity] = pfjet->neutralHadronMultiplicity();

        if (m_photonMultiplicity >= 0)
            buf[m_photonMultiplicity] = pfjet->photonMultiplicity();

        if (m_electronMultiplicity >= 0)
            buf[m_electronMultiplicity] = pfjet->electronMultiplicity();

        if (m_muonMultiplicity >= 0)
            buf[m_muonMultiplicity] = pfjet->muonMultiplicity();

        if (m_HFHadronMultiplicity >= 0)
            buf[m_HFHadronMultiplicity] = pfjet->HFHadronMultiplicity();

        if (m_HFEMMultiplicity >= 0)
            buf[m_HFEMMultiplicity] = pfjet->HFEMMultiplicity();

        if (m_chargedEmEnergyFraction >= 0)
            buf[m_chargedEmEnergyFraction] = pfjet->chargedEmEnergyFraction();

        if (m_chargedMuEnergyFraction >= 0)
            buf[m_chargedMuEnergyFraction] = pfjet->chargedMuEnergyFraction();

        if (m_neutralEmEnergyFraction >= 0)
            buf[m_neutralEmEnergyFraction] = pfjet->neutralEmEnergyFraction();

        if (m_EmEnergyFraction >= 0)
            buf[m_EmEnergyFraction] = pfjet->neutralEmEnergyFraction() +
                                      pfjet->chargedEmEnergyFraction();

        if (m_chargedMultiplicity >= 0)
            buf[m_chargedMultiplicity] = pfjet->chargedMultiplicity();

        if (m_neutralMultiplicity >= 0)
            buf[m_neutralMultiplicity] = pfjet->neutralMultiplicity();
    }
    else
    {
        // Not a particle flow jet
        if (m_chargedHadronEnergyFraction >= 0 ||
            m_neutralHadronEnergyFraction >= 0 ||
            m_photonEnergyFraction >= 0 ||
            m_electronEnergyFraction >= 0 ||
            m_muonEnergyFraction >= 0 ||
            m_HFHadronEnergyFraction >= 0 ||
            m_HFEMEnergyFraction >= 0 ||
            m_chargedHadronMultiplicity >= 0 ||
            m_neutralHadronMultiplicity >= 0 ||
            m_photonMultiplicity >= 0 ||
            m_electronMultiplicity >= 0 ||
            m_muonMultiplicity >= 0 ||
            m_HFHadronMultiplicity >= 0 ||
            m_HFEMMultiplicity >= 0 ||
            m_chargedEmEnergyFraction >= 0 ||
            m_chargedMuEnergyFraction >= 0 ||
            m_neutralEmEnergyFraction >= 0 ||
            m_EmEnergyFraction >= 0 ||
            m_chargedMultiplicity >= 0 ||
            m_neutralMultiplicity >= 0)
            throw cms::Exception("FFTJetBadConfig")
                << "In FFTGenericScaleCalculator::mapFFTJet: "
                << "this configuration is valid for particle flow jets only"
                << std::endl;
    }

    // Apply the scaling factors
    for (unsigned i=0; i<dim; ++i)
        buf[i] *= m_factors[i];
}
