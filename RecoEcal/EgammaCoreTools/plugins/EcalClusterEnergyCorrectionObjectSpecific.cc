/** \class EcalClusterEnergyCorrectionObjectSpecific
  *  Function that provides supercluster energy correction due to Bremsstrahlung loss
  *
  *  $Id: EcalClusterEnergyCorrectionObjectSpecific.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, October 2011
  */

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"

class EcalClusterEnergyCorrectionObjectSpecific : public EcalClusterFunctionBaseClass {
public:
  EcalClusterEnergyCorrectionObjectSpecific(const edm::ParameterSet &){};

  // get/set explicit methods for parameters
  const EcalClusterEnergyCorrectionObjectSpecificParameters *getParameters() const { return params_; }
  // check initialization
  void checkInit() const;

  // compute the correction
  float getValue(const reco::SuperCluster &, const int mode) const override;
  float getValue(const reco::BasicCluster &, const EcalRecHitCollection &) const override { return 0.; };

  // set parameters
  void init(const edm::EventSetup &es) override;

private:
  float fEta(float energy, float eta, int algorithm) const;
  float fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm) const;
  float fEt(float et, int algorithm) const;
  float fEnergy(float e, int algorithm) const;

  edm::ESHandle<EcalClusterEnergyCorrectionObjectSpecificParameters> esParams_;
  const EcalClusterEnergyCorrectionObjectSpecificParameters *params_;
};

void EcalClusterEnergyCorrectionObjectSpecific::init(const edm::EventSetup &es) {
  es.get<EcalClusterEnergyCorrectionObjectSpecificParametersRcd>().get(esParams_);
  params_ = esParams_.product();
}

void EcalClusterEnergyCorrectionObjectSpecific::checkInit() const {
  if (!params_) {
    // non-initialized function parameters: throw exception
    throw cms::Exception("EcalClusterEnergyCorrectionObjectSpecific::checkInit()")
        << "Trying to access an uninitialized correction function.\n"
           "Please call `init( edm::EventSetup &)' before any use of the function.\n";
  }
}

// Shower leakage corrections developed by Jungzhie et al. using TB data
// Developed for EB only!
float EcalClusterEnergyCorrectionObjectSpecific::fEta(float energy, float eta, int algorithm) const {
  //std::cout << "fEta function" << std::endl;

  // this correction is setup only for EB
  if (algorithm != 0)
    return energy;

  float ieta = fabs(eta) * (5 / 0.087);
  float p0 = (params_->params())[0];  // should be 40.2198
  float p1 = (params_->params())[1];  // should be -3.03103e-6

  //std::cout << "ieta=" << ieta << std::endl;

  float correctedEnergy = energy;
  if (ieta < p0)
    correctedEnergy = energy;
  else
    correctedEnergy = energy / (1.0 + p1 * (ieta - p0) * (ieta - p0));
  //std::cout << "ECEC fEta = " << correctedEnergy << std::endl;
  return correctedEnergy;
}

float EcalClusterEnergyCorrectionObjectSpecific::fBremEta(float sigmaPhiSigmaEta, float eta, int algorithm) const {
  const float etaCrackMin = 1.44;
  const float etaCrackMax = 1.56;

  //STD
  const int nBinsEta = 14;
  float leftEta[nBinsEta] = {0.02, 0.25, 0.46, 0.81, 0.91, 1.01, 1.16, etaCrackMax, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4};
  float rightEta[nBinsEta] = {0.25, 0.42, 0.77, 0.91, 1.01, 1.13, etaCrackMin, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4, 2.5};

  float xcorr[nBinsEta];

  float par0[nBinsEta];
  float par1[nBinsEta];
  float par2[nBinsEta];
  float par3[nBinsEta];
  float par4[nBinsEta];

  float sigmaPhiSigmaEtaMin = 0.8;
  float sigmaPhiSigmaEtaMax = 5.;

  float sigmaPhiSigmaEtaFit = -1;

  // extra protections
  // fix sigmaPhiSigmaEta boundaries
  if (sigmaPhiSigmaEta < sigmaPhiSigmaEtaMin)
    sigmaPhiSigmaEta = sigmaPhiSigmaEtaMin;
  if (sigmaPhiSigmaEta > sigmaPhiSigmaEtaMax)
    sigmaPhiSigmaEta = sigmaPhiSigmaEtaMax;

  // eta = 0
  if (std::abs(eta) < leftEta[0]) {
    eta = 0.02;
  }
  // outside acceptance
  if (std::abs(eta) >= rightEta[nBinsEta - 1]) {
    eta = 2.49;
  }  //if (DBG) std::cout << " WARNING [applyScCorrections]: std::abs(eta)  >=  rightEta[nBinsEta-1] " << std::endl;}

  int tmpEta = -1;
  for (int iEta = 0; iEta < nBinsEta; ++iEta) {
    if (leftEta[iEta] <= std::abs(eta) && std::abs(eta) < rightEta[iEta]) {
      tmpEta = iEta;
    }
  }

  if (algorithm == 0) {  //Electrons

    xcorr[0] = (params_->params())[2];
    xcorr[1] = (params_->params())[3];
    xcorr[2] = (params_->params())[4];
    xcorr[3] = (params_->params())[5];
    xcorr[4] = (params_->params())[6];
    xcorr[5] = (params_->params())[7];
    xcorr[6] = (params_->params())[8];
    xcorr[7] = (params_->params())[9];
    xcorr[8] = (params_->params())[10];
    xcorr[9] = (params_->params())[11];
    xcorr[10] = (params_->params())[12];
    xcorr[11] = (params_->params())[13];
    xcorr[12] = (params_->params())[14];
    xcorr[13] = (params_->params())[15];

    par0[0] = (params_->params())[16];
    par1[0] = (params_->params())[17];
    par2[0] = (params_->params())[18];
    par3[0] = (params_->params())[19];  //should be 0 (not used)
    par4[0] = (params_->params())[20];  //should be 0 (not used)

    par0[1] = (params_->params())[21];
    par1[1] = (params_->params())[22];
    par2[1] = (params_->params())[23];
    par3[1] = (params_->params())[24];
    par4[1] = (params_->params())[25];

    par0[2] = (params_->params())[26];
    par1[2] = (params_->params())[27];
    par2[2] = (params_->params())[28];
    par3[2] = (params_->params())[29];  //should be 0 (not used)
    par4[2] = (params_->params())[30];  //should be 0 (not used)

    par0[3] = (params_->params())[31];
    par1[3] = (params_->params())[32];
    par2[3] = (params_->params())[33];
    par2[4] = (params_->params())[34];  //should be 0 (not used)
    par2[5] = (params_->params())[35];  //should be 0 (not used)

    par0[4] = (params_->params())[36];
    par1[4] = (params_->params())[37];
    par2[4] = (params_->params())[38];
    par3[4] = (params_->params())[39];  //should be 0 (not used)
    par4[4] = (params_->params())[40];  //should be 0 (not used)

    par0[5] = (params_->params())[41];
    par1[5] = (params_->params())[42];
    par2[5] = (params_->params())[43];
    par3[5] = (params_->params())[44];  //should be 0 (not used)
    par4[5] = (params_->params())[45];  //should be 0 (not used)

    par0[6] = (params_->params())[46];
    par1[6] = (params_->params())[47];
    par2[6] = (params_->params())[48];
    par3[6] = (params_->params())[49];  //should be 0 (not used)
    par4[6] = (params_->params())[50];  //should be 0 (not used)

    par0[7] = (params_->params())[51];
    par1[7] = (params_->params())[52];
    par2[7] = (params_->params())[53];
    par3[7] = (params_->params())[54];  //should be 0 (not used)
    par4[7] = (params_->params())[55];  //should be 0 (not used)

    par0[8] = (params_->params())[56];
    par1[8] = (params_->params())[57];
    par2[8] = (params_->params())[58];
    par3[8] = (params_->params())[59];  //should be 0 (not used)
    par4[8] = (params_->params())[60];  //should be 0 (not used)

    par0[9] = (params_->params())[61];
    par1[9] = (params_->params())[62];
    par2[9] = (params_->params())[63];
    par3[9] = (params_->params())[64];  //should be 0 (not used)
    par4[9] = (params_->params())[65];  //should be 0 (not used)

    par0[10] = (params_->params())[66];
    par1[10] = (params_->params())[67];
    par2[10] = (params_->params())[68];
    par3[10] = (params_->params())[69];  //should be 0 (not used)
    par4[10] = (params_->params())[70];  //should be 0 (not used)

    par0[11] = (params_->params())[71];
    par1[11] = (params_->params())[72];
    par2[11] = (params_->params())[73];
    par3[11] = (params_->params())[74];  //should be 0 (not used)
    par4[11] = (params_->params())[75];  //should be 0 (not used)

    par0[12] = (params_->params())[76];
    par1[12] = (params_->params())[77];
    par2[12] = (params_->params())[78];
    par3[12] = (params_->params())[79];  //should be 0 (not used)
    par4[12] = (params_->params())[80];  //should be 0 (not used)

    par0[13] = (params_->params())[81];
    par1[13] = (params_->params())[82];
    par2[13] = (params_->params())[83];
    par3[13] = (params_->params())[84];  //should be 0 (not used)
    par4[13] = (params_->params())[85];  //should be 0 (not used)

    sigmaPhiSigmaEtaFit = 1.2;
  }

  if (algorithm == 1) {  //Photons

    xcorr[0] = (params_->params())[86];
    xcorr[1] = (params_->params())[87];
    xcorr[2] = (params_->params())[88];
    xcorr[3] = (params_->params())[89];
    xcorr[4] = (params_->params())[90];
    xcorr[5] = (params_->params())[91];
    xcorr[6] = (params_->params())[92];
    xcorr[7] = (params_->params())[93];
    xcorr[8] = (params_->params())[94];
    xcorr[9] = (params_->params())[95];
    xcorr[10] = (params_->params())[96];
    xcorr[11] = (params_->params())[97];
    xcorr[12] = (params_->params())[98];
    xcorr[13] = (params_->params())[99];

    par0[0] = (params_->params())[100];
    par1[0] = (params_->params())[101];
    par2[0] = (params_->params())[102];
    par3[0] = (params_->params())[103];
    par4[0] = (params_->params())[104];

    par0[1] = (params_->params())[105];
    par1[1] = (params_->params())[106];
    par2[1] = (params_->params())[107];
    par3[1] = (params_->params())[108];
    par4[1] = (params_->params())[109];

    par0[2] = (params_->params())[110];
    par1[2] = (params_->params())[111];
    par2[2] = (params_->params())[112];
    par3[2] = (params_->params())[113];
    par4[2] = (params_->params())[114];

    par0[3] = (params_->params())[115];
    par1[3] = (params_->params())[116];
    par2[3] = (params_->params())[117];
    par3[3] = (params_->params())[118];
    par4[3] = (params_->params())[119];

    par0[4] = (params_->params())[120];
    par1[4] = (params_->params())[121];
    par2[4] = (params_->params())[122];
    par3[4] = (params_->params())[123];
    par4[4] = (params_->params())[124];

    par0[5] = (params_->params())[125];
    par1[5] = (params_->params())[126];
    par2[5] = (params_->params())[127];
    par3[5] = (params_->params())[128];
    par4[5] = (params_->params())[129];

    par0[6] = (params_->params())[130];
    par1[6] = (params_->params())[131];
    par2[6] = (params_->params())[132];
    par3[6] = (params_->params())[133];
    par4[6] = (params_->params())[134];

    par0[7] = (params_->params())[135];
    par1[7] = (params_->params())[136];
    par2[7] = (params_->params())[137];
    par3[7] = (params_->params())[138];
    par4[7] = (params_->params())[139];

    par0[8] = (params_->params())[140];
    par1[8] = (params_->params())[141];
    par2[8] = (params_->params())[142];
    par3[8] = (params_->params())[143];
    par4[8] = (params_->params())[144];

    par0[9] = (params_->params())[145];
    par1[9] = (params_->params())[146];
    par2[9] = (params_->params())[147];
    par3[9] = (params_->params())[148];
    par4[9] = (params_->params())[149];

    par0[10] = (params_->params())[150];
    par1[10] = (params_->params())[151];
    par2[10] = (params_->params())[152];
    par3[10] = (params_->params())[153];
    par4[10] = (params_->params())[154];

    par0[11] = (params_->params())[155];
    par1[11] = (params_->params())[156];
    par2[11] = (params_->params())[157];
    par3[11] = (params_->params())[158];
    par4[11] = (params_->params())[159];

    par0[12] = (params_->params())[160];
    par1[12] = (params_->params())[161];
    par2[12] = (params_->params())[162];
    par3[12] = (params_->params())[163];
    par4[12] = (params_->params())[164];

    par0[13] = (params_->params())[165];
    par1[13] = (params_->params())[166];
    par2[13] = (params_->params())[167];
    par3[13] = (params_->params())[168];
    par4[13] = (params_->params())[169];

    sigmaPhiSigmaEtaFit = 1.;
  }

  // Interpolation
  float tmpInter = 1;
  // In eta cracks/gaps
  if (tmpEta == -1) {  // need to interpolate
    for (int iEta = 0; iEta < nBinsEta - 1; ++iEta) {
      if (rightEta[iEta] <= std::abs(eta) && std::abs(eta) < leftEta[iEta + 1]) {
        if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit) {
          if (algorithm == 0) {  //electron
            tmpInter = (par0[iEta] + sigmaPhiSigmaEta * par1[iEta] + sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[iEta] +
                        par0[iEta + 1] + sigmaPhiSigmaEta * par1[iEta + 1] +
                        sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[iEta + 1]) /
                       2.;
          }
          if (algorithm == 1) {  //photon
            tmpInter = (par0[iEta] * (1. - exp(-(sigmaPhiSigmaEta - par4[iEta]) / par1[iEta])) * par2[iEta] *
                            sigmaPhiSigmaEta +
                        par3[iEta] +
                        par0[iEta + 1] * (1. - exp(-(sigmaPhiSigmaEta - par4[iEta + 1]) / par1[iEta + 1])) *
                            par2[iEta + 1] * sigmaPhiSigmaEta +
                        par3[iEta + 1]) /
                       2.;
          }
        } else
          tmpInter = (xcorr[iEta] + xcorr[iEta + 1]) / 2.;
      }
    }
    return tmpInter;
  }

  if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit) {
    if (algorithm == 0)
      return par0[tmpEta] + sigmaPhiSigmaEta * par1[tmpEta] + sigmaPhiSigmaEta * sigmaPhiSigmaEta * par2[tmpEta];
    if (algorithm == 1)
      return par0[tmpEta] * (1. - exp(-(sigmaPhiSigmaEta - par4[tmpEta]) / par1[tmpEta])) * par2[tmpEta] *
                 sigmaPhiSigmaEta +
             par3[tmpEta];
  } else
    return xcorr[tmpEta];

  return 1.;
}

float EcalClusterEnergyCorrectionObjectSpecific::fEt(float ET, int algorithm) const {
  float par0 = -1;
  float par1 = -1;
  float par2 = -1;
  float par3 = -1;
  float par4 = -1;
  float par5 = -1;
  float par6 = -1;

  if (algorithm == 0) {  //Electrons EB

    par0 = (params_->params())[170];
    par1 = (params_->params())[171];
    par2 = (params_->params())[172];
    par3 = (params_->params())[173];
    par4 = (params_->params())[174];
    //assignments to 'par5'&'par6' have been deleted from here as they serve no purpose and cause dead assignment errors

    if (ET > 200)
      ET = 200;
    if (ET < 5)
      return 1.;
    if (5 <= ET && ET < 10)
      return par0;
    if (10 <= ET && ET <= 200)
      return (par1 + ET * par2) * (1 - par3 * exp(ET / par4));
  }

  if (algorithm == 1) {  //Electrons EE

    par0 = (params_->params())[177];
    par1 = (params_->params())[178];
    par2 = (params_->params())[179];
    par3 = (params_->params())[180];
    par4 = (params_->params())[181];
    //assignments to variables 'par5'&'par6' have been deleted from here as they serve no purpose and cause dead assignment errors

    if (ET > 200)
      ET = 200;
    if (ET < 5)
      return 1.;
    if (5 <= ET && ET < 10)
      return par0;
    if (10 <= ET && ET <= 200)
      return (par1 + ET * par2) * (1 - par3 * exp(ET / par4));
  }

  if (algorithm == 2) {  //Photons EB

    par0 = (params_->params())[184];
    par1 = (params_->params())[185];
    par2 = (params_->params())[186];
    par3 = (params_->params())[187];
    par4 = (params_->params())[188];
    //assignments to 'par5'&'par6' have been deleted from here as they serve no purpose and cause dead assignment errors

    if (ET < 5)
      return 1.;
    if (5 <= ET && ET < 10)
      return par0;
    if (10 <= ET && ET < 20)
      return par1;
    if (20 <= ET && ET < 140)
      return par2 + par3 * ET;
    if (140 <= ET)
      return par4;
  }

  if (algorithm == 3) {  //Photons EE

    par0 = (params_->params())[191];
    par1 = (params_->params())[192];
    par2 = (params_->params())[193];
    par3 = (params_->params())[194];
    par4 = (params_->params())[195];
    par5 = (params_->params())[196];
    par6 = (params_->params())[197];

    if (ET < 5)
      return 1.;
    if (5 <= ET && ET < 10)
      return par0;
    if (10 <= ET && ET < 20)
      return par1;
    if (20 <= ET && ET < 30)
      return par2;
    if (30 <= ET && ET < 200)
      return par3 + par4 * ET + par5 * ET * ET;
    if (200 <= ET)
      return par6;
  }

  return 1.;
}

float EcalClusterEnergyCorrectionObjectSpecific::fEnergy(float E, int algorithm) const {
  float par0 = -1;
  float par1 = -1;
  float par2 = -1;
  float par3 = -1;
  float par4 = -1;

  if (algorithm == 0) {  //Electrons EB
    return 1.;
  }

  if (algorithm == 1) {  //Electrons EE

    par0 = (params_->params())[198];
    par1 = (params_->params())[199];
    par2 = (params_->params())[200];
    par3 = (params_->params())[201];
    par4 = (params_->params())[202];

    if (E > par0)
      E = par0;
    if (E < 0)
      return 1.;
    if (0 <= E && E <= par0)
      return (par1 + E * par2) * (1 - par3 * exp(E / par4));
  }

  if (algorithm == 2) {  //Photons EB
    return 1.;
  }

  if (algorithm == 3) {  //Photons EE

    par0 = (params_->params())[203];
    par1 = (params_->params())[204];
    par2 = (params_->params())[205];
    //assignments to 'par3'&'par4' have been deleted from here as they serve no purpose and cause dead assignment errors

    if (E > par0)
      E = par0;
    if (E < 0)
      return 1.;
    if (0 <= E && E <= par0)
      return par1 + E * par2;
  }

  return 1.;
}

float EcalClusterEnergyCorrectionObjectSpecific::getValue(const reco::SuperCluster &superCluster,
                                                          const int mode) const {
  float corr = 1.;
  float corr2 = 1.;
  float energy = 0;

  int subdet = superCluster.seed()->hitsAndFractions()[0].first.subdetId();
  //std::cout << "subdet="<< subdet<< std::endl;

  //std::cout << "rawEnergy=" << superCluster.rawEnergy() << " SCeta=" << superCluster.eta() << std::endl;

  if (subdet == EcalBarrel) {
    float cetacorr = fEta(superCluster.rawEnergy(), superCluster.eta(), 0) / superCluster.rawEnergy();
    //std::cout << "cetacorr=" <<cetacorr<< std::endl;

    energy = superCluster.rawEnergy() * cetacorr;  //previously in CMSSW
    //energy = superCluster.rawEnergy()*fEta(e5x5, superCluster.seed()->eta(), 0)/e5x5;
  } else if (subdet == EcalEndcap) {
    energy = superCluster.rawEnergy() + superCluster.preshowerEnergy();
  }

  float newEnergy = energy;

  if (mode == 0) {  //Electron

    corr = fBremEta(superCluster.phiWidth() / superCluster.etaWidth(), superCluster.eta(), 0);

    float et = energy * std::sin(2 * std::atan(std::exp(-superCluster.eta()))) / corr;

    if (subdet == EcalBarrel)
      corr2 = corr * fEt(et, 0);
    if (subdet == EcalEndcap)
      corr2 = corr * fEnergy(energy / corr, 1);

    newEnergy = energy / corr2;
  }

  if (mode == 1) {  //low R9 Photons

    corr = fBremEta(superCluster.phiWidth() / superCluster.etaWidth(), superCluster.eta(), 1);

    float et = energy * std::sin(2 * std::atan(std::exp(-superCluster.eta()))) / corr;

    if (subdet == EcalBarrel)
      corr2 = corr * fEt(et, 2);
    if (subdet == EcalEndcap)
      corr2 = corr * fEnergy(energy / corr, 3);

    newEnergy = energy / corr2;
  }

  return newEnergy;
}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN(EcalClusterFunctionFactory,
                  EcalClusterEnergyCorrectionObjectSpecific,
                  "EcalClusterEnergyCorrectionObjectSpecific");
