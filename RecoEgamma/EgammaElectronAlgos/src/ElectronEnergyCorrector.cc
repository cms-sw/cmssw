#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EnergyUncertaintyElectronSpecific.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "TMath.h"

/****************************************************************************
 *
 * Classification based eta corrections for the ecal cluster energy
 *
 * \author Federico Ferri - INFN Milano, Bicocca university
 * \author Ivica Puljak - FESB, Split
 * \author Stephanie Baffioni - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 ****************************************************************************/

namespace {

  inline float energyError(float E, float* par) {
    return sqrt(pow(par[0] / sqrt(E), 2) + pow(par[1] / E, 2) + pow(par[2], 2));
  }

  // main correction function
  // new corrections: taken from EcalClusterCorrectionObjectSpecific.cc (N. Chanon et al.)
  // this is to prepare for class based corrections, for the time being the parameters are the same as for the SC corrections
  // code fully duplicated here, to be improved; electron means algorithm==0 and mode==0

  float fEta(float energy, float eta, int algorithm) {
    // corrections for electrons
    if (algorithm != 0) {
      edm::LogWarning("ElectronEnergyCorrector::fEta") << "algorithm should be 0 for electrons !";
      return energy;
    }
    //std::cout << "fEta function" << std::endl;

    // this correction is setup only for EB
    if (algorithm != 0)
      return energy;

    float ieta = fabs(eta) * (5 / 0.087);
    // bypass the DB reading for the time being
    //float p0 = (params_->params())[0];  // should be 40.2198
    //float p1 = (params_->params())[1];  // should be -3.03103e-6
    float p0 = 40.2198;
    float p1 = -3.03103e-6;

    //std::cout << "ieta=" << ieta << std::endl;

    float correctedEnergy = energy;
    if (ieta < p0)
      correctedEnergy = energy;
    else
      correctedEnergy = energy / (1.0 + p1 * (ieta - p0) * (ieta - p0));
    //std::cout << "ECEC fEta = " << correctedEnergy << std::endl;
    return correctedEnergy;
  }

  float fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm, reco::GsfElectron::Classification cl) {
    // corrections for electrons
    if (algorithm != 0) {
      edm::LogWarning("ElectronEnergyCorrector::fBremEta") << "algorithm should be 0 for electrons !";
      return 1.;
    }

    const float etaCrackMin = 1.44;
    const float etaCrackMax = 1.56;

    //STD
    const int nBinsEta = 14;
    float leftEta[nBinsEta] = {0.02, 0.25, 0.46, 0.81, 0.91, 1.01, 1.16, etaCrackMax, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4};
    float rightEta[nBinsEta] = {0.25, 0.42, 0.77, 0.91, 1.01, 1.13, etaCrackMin, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4, 2.5};

    // eta = 0
    if (TMath::Abs(eta) < leftEta[0]) {
      eta = 0.02;
    }

    // outside acceptance
    if (TMath::Abs(eta) >= rightEta[nBinsEta - 1]) {
      eta = 2.49;
    }  //if (DBG) std::cout << " WARNING [applyScCorrections]: TMath::Abs(eta)  >=  rightEta[nBinsEta-1] " << std::endl;}

    int tmpEta = -1;
    for (int iEta = 0; iEta < nBinsEta; ++iEta) {
      if (leftEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) < rightEta[iEta]) {
        tmpEta = iEta;
      }
    }

    float xcorr[nBinsEta][reco::GsfElectron::GAP + 1] = {{1.00227, 1.00227, 1.00227, 1.00227, 1.00227},
                                                         {1.00252, 1.00252, 1.00252, 1.00252, 1.00252},
                                                         {1.00225, 1.00225, 1.00225, 1.00225, 1.00225},
                                                         {1.00159, 1.00159, 1.00159, 1.00159, 1.00159},
                                                         {0.999475, 0.999475, 0.999475, 0.999475, 0.999475},
                                                         {0.997203, 0.997203, 0.997203, 0.997203, 0.997203},
                                                         {0.993886, 0.993886, 0.993886, 0.993886, 0.993886},
                                                         {0.971262, 0.971262, 0.971262, 0.971262, 0.971262},
                                                         {0.975922, 0.975922, 0.975922, 0.975922, 0.975922},
                                                         {0.979087, 0.979087, 0.979087, 0.979087, 0.979087},
                                                         {0.98495, 0.98495, 0.98495, 0.98495, 0.98495},
                                                         {0.98781, 0.98781, 0.98781, 0.98781, 0.98781},
                                                         {0.989546, 0.989546, 0.989546, 0.989546, 0.989546},
                                                         {0.989638, 0.989638, 0.989638, 0.989638, 0.989638}};

    float par0[nBinsEta][reco::GsfElectron::GAP + 1] = {{0.994949, 1.00718, 1.00718, 1.00556, 1.00718},
                                                        {1.009, 1.00713, 1.00713, 1.00248, 1.00713},
                                                        {0.999395, 1.00641, 1.00641, 1.00293, 1.00641},
                                                        {0.988662, 1.00761, 1.001, 0.99972, 1.00761},
                                                        {0.998443, 1.00682, 1.001, 1.00282, 1.00682},
                                                        {1.00285, 1.0073, 1.001, 1.00396, 1.0073},
                                                        {0.993053, 1.00462, 1.01341, 1.00184, 1.00462},
                                                        {1.10561, 0.972798, 1.02835, 0.995218, 0.972798},
                                                        {0.893741, 0.981672, 0.98982, 1.01712, 0.981672},
                                                        {0.911123, 0.98251, 1.03466, 1.00824, 0.98251},
                                                        {0.981931, 0.986123, 0.954295, 1.0202, 0.986123},
                                                        {0.905634, 0.990124, 0.928934, 0.998492, 0.990124},
                                                        {0.919343, 0.990187, 0.967526, 0.963923, 0.990187},
                                                        {0.844783, 0.99372, 0.923808, 0.953001, 0.99372}};

    float par1[nBinsEta][reco::GsfElectron::GAP + 1] = {
        {0.0111034, -0.00187886, -0.00187886, -0.00289304, -0.00187886},
        {-0.00969012, -0.00227574, -0.00227574, -0.00182187, -0.00227574},
        {0.00389454, -0.00259935, -0.00259935, -0.00211059, -0.00259935},
        {0.017095, -0.00433692, -0.00302335, -0.00241385, -0.00433692},
        {-0.00049009, -0.00551324, -0.00302335, -0.00532352, -0.00551324},
        {-0.00252723, -0.00799669, -0.00302335, -0.00823109, -0.00799669},
        {0.00332567, -0.00870057, -0.0170581, -0.011482, -0.00870057},
        {-0.213285, -0.000771577, -0.036007, -0.0187526, -0.000771577},
        {0.1741, -0.00202028, -0.0233995, -0.0302066, -0.00202028},
        {0.152794, 0.00441308, -0.0468563, -0.0158817, 0.00441308},
        {0.0351465, 0.00832913, 0.0358028, -0.0233262, 0.00832913},
        {0.185781, 0.00742879, 0.08858, 0.00568078, 0.00742879},
        {0.153088, 0.0094608, 0.0489979, 0.0491897, 0.0094608},
        {0.296681, 0.00560406, 0.106492, 0.0652007, 0.00560406}};

    float par2[nBinsEta][reco::GsfElectron::GAP + 1] = {{-0.00330844, 0, 0, 5.62441e-05, 0},
                                                        {0.00329373, 0, 0, -0.000113883, 0},
                                                        {-0.00104661, 0, 0, -0.000152794, 0},
                                                        {-0.0060409, 0, -0.000257724, -0.000202099, 0},
                                                        {-0.000742866, 0, -0.000257724, -2.06003e-05, 0},
                                                        {-0.00205425, 0, -0.000257724, 3.84179e-05, 0},
                                                        {-0.00350757, 0, 0.000590483, 0.000323723, 0},
                                                        {0.0794596, -0.00276696, 0.00205854, 0.000356716, -0.00276696},
                                                        {-0.092436, -0.00471028, 0.00062096, 0.00088347, -0.00471028},
                                                        {-0.0855029, -0.00809139, 0.00284102, -0.00366903, -0.00809139},
                                                        {-0.0306209, -0.00944584, -0.0145892, -0.00176969, -0.00944584},
                                                        {-0.0996414, -0.00960462, -0.0328264, -0.00983844, -0.00960462},
                                                        {-0.0784107, -0.010172, -0.0256722, -0.0215133, -0.010172},
                                                        {-0.145815, -0.00943169, -0.0414525, -0.027087, -0.00943169}};

    float sigmaPhiSigmaEtaMin[reco::GsfElectron::GAP + 1] = {0.8, 0.8, 0.8, 0.8, 0.8};
    float sigmaPhiSigmaEtaMax[reco::GsfElectron::GAP + 1] = {5., 5., 5., 5., 5.};
    float sigmaPhiSigmaEtaFit[reco::GsfElectron::GAP + 1] = {1.2, 1.2, 1.2, 1.2, 1.2};

    // extra protections
    // fix sigmaPhiSigmaEta boundaries
    if (sigmaPhiSigmaEta < sigmaPhiSigmaEtaMin[cl]) {
      sigmaPhiSigmaEta = sigmaPhiSigmaEtaMin[cl];
    }
    if (sigmaPhiSigmaEta > sigmaPhiSigmaEtaMax[cl]) {
      sigmaPhiSigmaEta = sigmaPhiSigmaEtaMax[cl];
    }

    // In eta cracks/gaps
    if (tmpEta == -1)  // need to interpolate
    {
      float tmpInter = 1;
      for (int iEta = 0; iEta < nBinsEta - 1; ++iEta) {
        if (rightEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) < leftEta[iEta + 1]) {
          if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit[cl]) {
            tmpInter =
                (par0[iEta][cl] + sigmaPhiSigmaEta * par1[iEta][cl] +
                 sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[iEta][cl] + par0[iEta + 1][cl] +
                 sigmaPhiSigmaEta * par1[iEta + 1][cl] + sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[iEta + 1][cl]) /
                2.;
          } else
            tmpInter = (xcorr[iEta][cl] + xcorr[iEta + 1][cl]) / 2.;
        }
      }
      return tmpInter;
    }

    if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit[cl]) {
      return par0[tmpEta][cl] + sigmaPhiSigmaEta * par1[tmpEta][cl] +
             sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[tmpEta][cl];
    } else {
      return xcorr[tmpEta][cl];
    }

    return 1.;
  }

  float fEt(float ET, int algorithm, reco::GsfElectron::Classification cl) {
    if (algorithm == 0)  //Electrons EB
    {
      const float parClassIndep[5] = {0.97213, 0.999528, 5.61192e-06, 0.0143269, -17.1776};
      const float par[reco::GsfElectron::GAP + 1][5] = {
          {0.974327, 0.996127, 5.99401e-05, 0.159813, -3.80392},
          {0.97213, 0.999528, 5.61192e-06, 0.0143269, -17.1776},
          {0.940666, 0.988894, 0.00017474, 0.25603, -4.58153},
          {0.969526, 0.98572, 0.000193842, 4.21548, -1.37159},
          {parClassIndep[0], parClassIndep[1], parClassIndep[2], parClassIndep[3], parClassIndep[4]}};
      if (ET > 200) {
        ET = 200;
      }
      if (ET > 100) {
        return (parClassIndep[1] + ET * parClassIndep[2]) * (1 - parClassIndep[3] * exp(ET / parClassIndep[4]));
      }
      if (ET >= 10) {
        return (par[cl][1] + ET * par[cl][2]) * (1 - par[cl][3] * exp(ET / par[cl][4]));
      }
      if (ET >= 5) {
        return par[cl][0];
      }
      return 1.;
    } else if (algorithm == 1)  //Electrons EE
    {
      float par[reco::GsfElectron::GAP + 1][5] = {{0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461},
                                                  {0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461},
                                                  {0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461},
                                                  {0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461},
                                                  {0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461}};
      if (ET > 200) {
        ET = 200;
      }
      if (ET < 5) {
        return 1.;
      }
      if (5 <= ET && ET < 10) {
        return par[cl][0];
      }
      if (10 <= ET && ET <= 200) {
        return (par[cl][1] + ET * par[cl][2]) * (1 - par[cl][3] * exp(ET / par[cl][4]));
      }
    } else {
      edm::LogWarning("ElectronEnergyCorrector::fEt") << "algorithm should be 0 or 1 for electrons !";
    }
    return 1.;
  }

  float fEnergy(float E, int algorithm, reco::GsfElectron::Classification cl) {
    if (algorithm == 0)  // Electrons EB
    {
      return 1.;
    } else if (algorithm == 1)  // Electrons EE
    {
      float par0[reco::GsfElectron::GAP + 1] = {400, 400, 400, 400, 400};
      float par1[reco::GsfElectron::GAP + 1] = {0.999545, 0.982475, 0.986217, 0.996763, 0.982475};
      float par2[reco::GsfElectron::GAP + 1] = {1.26568e-05, 4.95413e-05, 5.02161e-05, 2.8485e-06, 4.95413e-05};
      float par3[reco::GsfElectron::GAP + 1] = {0.0696757, 0.16886, 0.115317, 0.12098, 0.16886};
      float par4[reco::GsfElectron::GAP + 1] = {-54.3468, -30.1517, -26.3805, -62.0538, -30.1517};

      if (E > par0[cl]) {
        E = par0[cl];
      }
      if (E < 0) {
        return 1.;
      }
      if (0 <= E && E <= par0[cl]) {
        return (par1[cl] + E * par2[cl]) * (1 - par3[cl] * exp(E / par4[cl]));
      }
    } else {
      edm::LogWarning("ElectronEnergyCorrector::fEnergy") << "algorithm should be 0 or 1 for electrons !";
    }
    return 1.;
  }

}  // namespace

double egamma::classBasedElectronEnergyUncertainty(reco::GsfElectron const& electron) {
  double ecalEnergy = electron.correctedEcalEnergy();
  double eleEta = electron.superCluster()->eta();
  double brem = electron.superCluster()->etaWidth() / electron.superCluster()->phiWidth();
  return egamma::electronEnergyUncertainty(electron.classification(), eleEta, brem, ecalEnergy);
}

double egamma::simpleElectronEnergyUncertainty(reco::GsfElectron const& electron) {
  double error = 999.;
  double ecalEnergy = electron.correctedEcalEnergy();

  if (electron.isEB()) {
    float parEB[3] = {5.24e-02, 2.01e-01, 1.00e-02};
    error = ecalEnergy * energyError(ecalEnergy, parEB);
  } else if (electron.isEE()) {
    float parEE[3] = {1.46e-01, 9.21e-01, 1.94e-03};
    error = ecalEnergy * energyError(ecalEnergy, parEE);
  } else {
    edm::LogWarning("ElectronEnergyCorrector::simpleParameterizationUncertainty")
        << "nor barrel neither endcap electron !";
  }

  return error;
}

float egamma::classBasedElectronEnergy(reco::GsfElectron const& electron,
                                       reco::BeamSpot const& bs,
                                       EcalClusterFunctionBaseClass const& crackCorrectionFunction) {
  auto elClass = electron.classification();

  // new corrections from N. Chanon et al., taken from EcalClusterCorrectionObjectSpecific.cc
  float corr = 1.;
  float corr2 = 1.;
  float energy = electron.superCluster()->energy();

  //int subdet = electron.superCluster()->seed()->hitsAndFractions()[0].first.subdetId();

  if (electron.isEB()) {
    float cetacorr = fEta(electron.superCluster()->rawEnergy(), electron.superCluster()->eta(), 0) /
                     electron.superCluster()->rawEnergy();
    energy = electron.superCluster()->rawEnergy() * cetacorr;  //previously in CMSSW
    //energy = superCluster.rawEnergy()*fEta(e5x5, superCluster.seed()->eta(), 0)/e5x5;
  } else if (electron.isEE()) {
    energy = electron.superCluster()->rawEnergy() + electron.superCluster()->preshowerEnergy();
  } else {
    edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")
        << "nor barrel neither endcap electron !";
  }

  corr = fBremEta(electron.superCluster()->phiWidth() / electron.superCluster()->etaWidth(),
                  electron.superCluster()->eta(),
                  0,
                  elClass);

  float et = energy * TMath::Sin(2 * TMath::ATan(TMath::Exp(-electron.superCluster()->eta()))) / corr;

  if (electron.isEB()) {
    corr2 = corr * fEt(et, 0, elClass);
  } else if (electron.isEE()) {
    corr2 = corr * fEnergy(energy / corr, 1, elClass);
  } else {
    edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")
        << "nor barrel neither endcap electron !";
  }

  float newEnergy = energy / corr2;

  // cracks
  double crackcor = 1.;
  for (reco::CaloCluster_iterator cIt = electron.superCluster()->clustersBegin();
       cIt != electron.superCluster()->clustersEnd();
       ++cIt) {
    const reco::CaloClusterPtr cc = *cIt;
    crackcor *= (electron.superCluster()->rawEnergy() + cc->energy() * (crackCorrectionFunction.getValue(*cc) - 1.)) /
                electron.superCluster()->rawEnergy();
  }
  newEnergy *= crackcor;

  return newEnergy;
}
