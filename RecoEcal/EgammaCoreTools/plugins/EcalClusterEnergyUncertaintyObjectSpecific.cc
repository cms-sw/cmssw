/** \class EcalClusterEnergyUncertainty
  *  Function that provides uncertainty on supercluster energy measurement
  *  Available numbers: total effective uncertainty (in GeV)
  *                     assymetric uncertainties (positive and negative)
  *
  *  $Id: EcalClusterEnergyUncertainty.h
  *  $Date:
  *  $Revision:
  *  \author Nicolas Chanon, December 2011
  */

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "FWCore/Framework/interface/EventSetup.h"

class EcalClusterEnergyUncertaintyObjectSpecific : public EcalClusterFunctionBaseClass {
public:
  EcalClusterEnergyUncertaintyObjectSpecific(const edm::ParameterSet &){};

  // check initialization
  void checkInit() const {}

  // compute the correction
  float getValue(const reco::SuperCluster &, const int mode) const override;
  float getValue(const reco::BasicCluster &, const EcalRecHitCollection &) const override { return 0.; };

  // set parameters
  void init(const edm::EventSetup &es) override {}
};

float EcalClusterEnergyUncertaintyObjectSpecific::getValue(const reco::SuperCluster &superCluster,
                                                           const int mode) const {
  checkInit();

  // mode  = 0 returns electron energy uncertainty

  float en = superCluster.energy();
  float eta = fabs(superCluster.eta());
  float et = en / cosh(eta);
  float brem = superCluster.etaWidth() != 0 ? superCluster.phiWidth() / superCluster.etaWidth() : 0;

  const int nBinsEta = 6;
  const float EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  const int nBinsBrem = 6;
  const float BremBins[nBinsBrem + 1] = {0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0] = 0.00640519;
  par1[0][0] = 0.257578;
  par2[0][0] = 1.72437;
  par3[0][0] = 4.04686e-06;

  par0[0][1] = 0.00709569;
  par1[0][1] = 0.279844;
  par2[0][1] = 1.13789;
  par3[0][1] = 1.16239e-05;

  par0[0][2] = 0.0075544;
  par1[0][2] = 0.341346;
  par2[0][2] = 0.513396;
  par3[0][2] = 2.90054e-06;

  par0[0][3] = 0.00659365;
  par1[0][3] = 0.517649;
  par2[0][3] = -3.1847;
  par3[0][3] = 7.37152e-07;

  par0[0][4] = 0.00771696;
  par1[0][4] = 0.492897;
  par2[0][4] = -1.42222;
  par3[0][4] = 0.000358677;

  par0[0][5] = 0.00561532;
  par1[0][5] = 0.655138;
  par2[0][5] = -3.29839;
  par3[0][5] = 6.25898e-07;

  par0[1][0] = 0.00273646;
  par1[1][0] = 0.714568;
  par2[1][0] = -4.82956;
  par3[1][0] = 4.45878e-07;

  par0[1][1] = 0.00679797;
  par1[1][1] = 0.472856;
  par2[1][1] = -0.281699;
  par3[1][1] = 5.46479e-05;

  par0[1][2] = 0.00845532;
  par1[1][2] = 0.611624;
  par2[1][2] = -1.10104;
  par3[1][2] = 1.16803e-05;

  par0[1][3] = 0.00831068;
  par1[1][3] = 0.853653;
  par2[1][3] = -4.23761;
  par3[1][3] = 2.61247e-05;

  par0[1][4] = 0.00845457;
  par1[1][4] = 0.984985;
  par2[1][4] = -5.19548;
  par3[1][4] = 2.05044e-07;

  par0[1][5] = 0.0110227;
  par1[1][5] = 1.00356;
  par2[1][5] = -4.31936;
  par3[1][5] = 0.14384;

  par0[2][0] = -0.00192618;
  par1[2][0] = 1.69986;
  par2[2][0] = -16.4355;
  par3[2][0] = 1.94946e-06;

  par0[2][1] = 0.0067622;
  par1[2][1] = 0.792209;
  par2[2][1] = -1.18521;
  par3[2][1] = 0.066577;

  par0[2][2] = 0.00761595;
  par1[2][2] = 1.03058;
  par2[2][2] = -4.17237;
  par3[2][2] = 0.168543;

  par0[2][3] = 0.0119179;
  par1[2][3] = 0.910145;
  par2[2][3] = -2.14122;
  par3[2][3] = 0.00342264;

  par0[2][4] = 0.0139921;
  par1[2][4] = 1.01488;
  par2[2][4] = -2.46637;
  par3[2][4] = 0.0458434;

  par0[2][5] = 0.013724;
  par1[2][5] = 1.49078;
  par2[2][5] = -6.60661;
  par3[2][5] = 0.297821;

  par0[3][0] = -0.00197909;
  par1[3][0] = 4.40696;
  par2[3][0] = -4.88737;
  par3[3][0] = 4.99999;

  par0[3][1] = 0.0340196;
  par1[3][1] = 3.86278;
  par2[3][1] = -10.899;
  par3[3][1] = 0.130098;

  par0[3][2] = 0.0102397;
  par1[3][2] = 8.99643;
  par2[3][2] = -31.5122;
  par3[3][2] = 0.00118335;

  par0[3][3] = 0.0110891;
  par1[3][3] = 8.01794;
  par2[3][3] = -21.9038;
  par3[3][3] = 0.000245975;

  par0[3][4] = 0.0328931;
  par1[3][4] = 4.73441;
  par2[3][4] = -12.1148;
  par3[3][4] = 3.01721e-05;

  par0[3][5] = 0.0395614;
  par1[3][5] = 3.54327;
  par2[3][5] = -12.6514;
  par3[3][5] = 0.119761;

  par0[4][0] = 0.0121809;
  par1[4][0] = 0.965608;
  par2[4][0] = -4.19667;
  par3[4][0] = 0.129896;

  par0[4][1] = 0.0168951;
  par1[4][1] = 1.0218;
  par2[4][1] = -4.03078;
  par3[4][1] = 0.374291;

  par0[4][2] = 0.0213549;
  par1[4][2] = 1.29613;
  par2[4][2] = -4.89024;
  par3[4][2] = 0.0297165;

  par0[4][3] = 0.0262602;
  par1[4][3] = 1.41674;
  par2[4][3] = -5.94928;
  par3[4][3] = 0.19298;

  par0[4][4] = 0.0334892;
  par1[4][4] = 1.48572;
  par2[4][4] = -5.3175;
  par3[4][4] = 0.0157013;

  par0[4][5] = 0.0347093;
  par1[4][5] = 1.63127;
  par2[4][5] = -7.27426;
  par3[4][5] = 0.201164;

  par0[5][0] = 0.0185321;
  par1[5][0] = 0.255205;
  par2[5][0] = 1.56798;
  par3[5][0] = 5.07655e-11;

  par0[5][1] = 0.0182718;
  par1[5][1] = 0.459086;
  par2[5][1] = -0.48198;
  par3[5][1] = 0.00114946;

  par0[5][2] = 0.0175505;
  par1[5][2] = 0.92848;
  par2[5][2] = -4.52737;
  par3[5][2] = 0.154827;

  par0[5][3] = 0.0233833;
  par1[5][3] = 0.804105;
  par2[5][3] = -3.75131;
  par3[5][3] = 2.84172;

  par0[5][4] = 0.0334892;
  par1[5][4] = 1.48572;
  par2[5][4] = -5.3175;
  par3[5][4] = 0.0157013;

  par0[5][5] = 0.0347093;
  par1[5][5] = 1.63127;
  par2[5][5] = -7.27426;
  par3[5][5] = 0.201164;

  int iEtaSl = -1;
  for (int iEta = 0; iEta < nBinsEta; ++iEta) {
    if (EtaBins[iEta] <= eta && eta < EtaBins[iEta + 1]) {
      iEtaSl = iEta;
    }
  }

  int iBremSl = -1;
  for (int iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
    if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
      iBremSl = iBrem;
    }
  }

  //this code is confusing as it has no concept of under and overflow bins
  //we will use Et as an example but also applies to eta
  //underflow is 1st bin (naively currently labeled as 0 to 0.7, its really <0.7)
  //overflow is the final bin (naviely currently labeled as 5 to 10, its really >=5)
  //logic: if brem is 0<=brem <0.7 it will be already set to the 1st bin, this checks if its <0
  //logic: if brem is 5<=brem<10 it will be set to the last bin so this then checks if its >5 at which point
  //it also assigns it to the last bin. The value of 5 will have already been assigned in the for
  //loop above to the last bin so its okay that its a >5 test
  if (eta > EtaBins[nBinsEta - 1])
    iEtaSl = nBinsEta - 1;
  if (brem < BremBins[0])
    iBremSl = 0;
  if (brem > BremBins[nBinsBrem - 1])
    iBremSl = nBinsBrem - 1;

  float uncertainty = 0;
  if (et <= 5)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
  if (et >= 200)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (200 - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((200 - par2[iEtaSl][iBremSl]) * (200 - par2[iEtaSl][iBremSl]));

  if (et > 5 && et < 200)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));

  return (uncertainty * en);
}

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
DEFINE_EDM_PLUGIN(EcalClusterFunctionFactory,
                  EcalClusterEnergyUncertaintyObjectSpecific,
                  "EcalClusterEnergyUncertaintyObjectSpecific");
