#include "RecoEgamma/EgammaPhotonAlgos/interface/EnergyUncertaintyPhotonSpecific.h"

#include <iostream>

EnergyUncertaintyPhotonSpecific::EnergyUncertaintyPhotonSpecific(const edm::ParameterSet& config) {}

EnergyUncertaintyPhotonSpecific::~EnergyUncertaintyPhotonSpecific() {}

void EnergyUncertaintyPhotonSpecific::init(const edm::EventSetup& theEventSetup) {}

double EnergyUncertaintyPhotonSpecific::computePhotonEnergyUncertainty_lowR9(double eta, double brem, double energy) {
  double et = energy / cosh(eta);

  const int nBinsEta = 6;
  const double EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  const int nBinsBrem = 6;
  const double BremBins[nBinsBrem + 1] = {0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0] = 0.0232291;
  par1[0][0] = 0;
  par2[0][0] = 0;
  par3[0][0] = 0;

  par0[0][1] = 0.00703187;
  par1[0][1] = 0.646644;
  par2[0][1] = -7.4698;
  par3[0][1] = 5.53373e-08;

  par0[0][2] = 0.00692465;
  par1[0][2] = 0.292698;
  par2[0][2] = 4.16907;
  par3[0][2] = 5.61149e-06;

  par0[0][3] = 0.00855993;
  par1[0][3] = 0.280843;
  par2[0][3] = 4.25527;
  par3[0][3] = 9.6404e-07;

  par0[0][4] = 0.00795058;
  par1[0][4] = 0.370007;
  par2[0][4] = 3.03429;
  par3[0][4] = 4.43986e-07;

  par0[0][5] = 0.0107494;
  par1[0][5] = 0.276159;
  par2[0][5] = 4.44532;
  par3[0][5] = 2.58822e-06;

  par0[1][0] = 0.0614866;
  par1[1][0] = 0;
  par2[1][0] = 0;
  par3[1][0] = 0;

  par0[1][1] = 0.00894211;
  par1[1][1] = 0.466937;
  par2[1][1] = 3.33434;
  par3[1][1] = 0.000114835;

  par0[1][2] = 0.0102959;
  par1[1][2] = 0.313568;
  par2[1][2] = 6.34301;
  par3[1][2] = 2.86726e-07;

  par0[1][3] = 0.0128934;
  par1[1][3] = 0.302943;
  par2[1][3] = 6.35598;
  par3[1][3] = 0.00190694;

  par0[1][4] = 0.0130199;
  par1[1][4] = 0.505135;
  par2[1][4] = 2.52964;
  par3[1][4] = 0.120204;

  par0[1][5] = 0.0180839;
  par1[1][5] = 0.382134;
  par2[1][5] = 5.3388;
  par3[1][5] = 3.59921e-07;

  par0[2][0] = 0.0291343;
  par1[2][0] = 0;
  par2[2][0] = 0;
  par3[2][0] = 0;

  par0[2][1] = 0.00876269;
  par1[2][1] = 0.375159;
  par2[2][1] = 7.11411;
  par3[2][1] = 0.0438575;

  par0[2][2] = 0.0120863;
  par1[2][2] = 0.397635;
  par2[2][2] = 5.97451;
  par3[2][2] = 0.0469782;

  par0[2][3] = 0.0112655;
  par1[2][3] = 0.856565;
  par2[2][3] = -5.76122;
  par3[2][3] = 4.99993;

  par0[2][4] = 0.0168267;
  par1[2][4] = 0.636468;
  par2[2][4] = -1.54548;
  par3[2][4] = 4.99992;

  par0[2][5] = 0.0168059;
  par1[2][5] = 1.09268;
  par2[2][5] = -0.547554;
  par3[2][5] = 0.0952985;

  par0[3][0] = 0.158403;
  par1[3][0] = 0;
  par2[3][0] = 0;
  par3[3][0] = 0;

  par0[3][1] = 0.0717431;
  par1[3][1] = 1.66981;
  par2[3][1] = 6.86275;
  par3[3][1] = 0.00543544;

  par0[3][2] = 0.0385666;
  par1[3][2] = 3.6319;
  par2[3][2] = -3.76633;
  par3[3][2] = 6.56718e-05;

  par0[3][3] = 0.0142631;
  par1[3][3] = 8.85991;
  par2[3][3] = -32.6073;
  par3[3][3] = 0.00119538;

  par0[3][4] = 0.0421638;
  par1[3][4] = 3.1289;
  par2[3][4] = -6.58653;
  par3[3][4] = 1.10125e-05;

  par0[3][5] = 0.046331;
  par1[3][5] = 1.29951;
  par2[3][5] = 1.76117;
  par3[3][5] = 0.00204206;

  par0[4][0] = 0.0483944;
  par1[4][0] = 0;
  par2[4][0] = 0;
  par3[4][0] = 0;

  par0[4][1] = 0.0168516;
  par1[4][1] = 1.19617;
  par2[4][1] = -6.78666;
  par3[4][1] = 4.98192;

  par0[4][2] = 0.0243039;
  par1[4][2] = 0.994626;
  par2[4][2] = -4.26073;
  par3[4][2] = 4.99984;

  par0[4][3] = 0.031795;
  par1[4][3] = 0.875925;
  par2[4][3] = 1.43183;
  par3[4][3] = 0.0920944;

  par0[4][4] = 0.0414953;
  par1[4][4] = 0.654605;
  par2[4][4] = 4.45367;
  par3[4][4] = 0.030385;

  par0[4][5] = 0.058031;
  par1[4][5] = 0.292915;
  par2[4][5] = 8.48307;
  par3[4][5] = 0.0134321;

  par0[5][0] = 0.107158;
  par1[5][0] = 0;
  par2[5][0] = 0;
  par3[5][0] = 0;

  par0[5][1] = 0.021685;
  par1[5][1] = 0.574207;
  par2[5][1] = -0.566981;
  par3[5][1] = 0.0120609;

  par0[5][2] = 0.0196619;
  par1[5][2] = 0.940217;
  par2[5][2] = -6.05845;
  par3[5][2] = 0.000193818;

  par0[5][3] = 0.0324734;
  par1[5][3] = 0.574766;
  par2[5][3] = -5.23571;
  par3[5][3] = 4.9419;

  par0[5][4] = 0.0414953;
  par1[5][4] = 0.654605;
  par2[5][4] = 4.45367;
  par3[5][4] = 0.030385;

  par0[5][5] = 0.058031;
  par1[5][5] = 0.292915;
  par2[5][5] = 8.48307;
  par3[5][5] = 0.0134321;

  int iEtaSl = -1;
  for (int iEta = 0; iEta < nBinsEta; ++iEta) {
    if (EtaBins[iEta] <= std::abs(eta) && std::abs(eta) < EtaBins[iEta + 1]) {
      iEtaSl = iEta;
    }
  }

  int iBremSl = -1;
  for (int iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
    if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
      iBremSl = iBrem;
    }
  }

  if (std::abs(eta) > 2.5)
    iEtaSl = nBinsEta - 1;
  if (brem < BremBins[0])
    iBremSl = 0;
  if (brem > BremBins[nBinsBrem - 1])
    iBremSl = nBinsBrem - 1;

  float uncertainty = 0;
  if (et < 5)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
  if (et > 200)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (200 - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((200 - par2[iEtaSl][iBremSl]) * (200 - par2[iEtaSl][iBremSl]));

  if (et > 5 && et < 200)
    uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                  par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));

  return (uncertainty * energy);
}

double EnergyUncertaintyPhotonSpecific::computePhotonEnergyUncertainty_highR9(double eta, double brem, double energy) {
  double et = energy / cosh(eta);

  const int nBinsEta = 6;
  const double EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  const int nBinsBrem = 6;
  const double BremBins[nBinsBrem + 1] = {0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0] = 0.00806753;
  par1[0][0] = 0.143754;
  par2[0][0] = -0.00368104;
  par3[0][0] = 0.219829;

  par0[0][1] = 0.00899298;
  par1[0][1] = 0.10159;
  par2[0][1] = 4.70884;
  par3[0][1] = 9.07419e-08;

  par0[0][2] = 0;
  par1[0][2] = 0;
  par2[0][2] = 0;
  par3[0][2] = 0;

  par0[0][3] = 0;
  par1[0][3] = 0;
  par2[0][3] = 0;
  par3[0][3] = 0;

  par0[0][4] = 0;
  par1[0][4] = 0;
  par2[0][4] = 0;
  par3[0][4] = 0;

  par0[0][5] = 0;
  par1[0][5] = 0;
  par2[0][5] = 0;
  par3[0][5] = 0;

  par0[1][0] = 0.00880649;
  par1[1][0] = 0.0716169;
  par2[1][0] = 5.23856;
  par3[1][0] = 0.00632907;

  par0[1][1] = 0.00972275;
  par1[1][1] = 0.0752675;
  par2[1][1] = 3.35623;
  par3[1][1] = 2.49397e-07;

  par0[1][2] = 0;
  par1[1][2] = 0;
  par2[1][2] = 0;
  par3[1][2] = 0;

  par0[1][3] = 0;
  par1[1][3] = 0;
  par2[1][3] = 0;
  par3[1][3] = 0;

  par0[1][4] = 0;
  par1[1][4] = 0;
  par2[1][4] = 0;
  par3[1][4] = 0;

  par0[1][5] = 0;
  par1[1][5] = 0;
  par2[1][5] = 0;
  par3[1][5] = 0;

  par0[2][0] = 0.0101474;
  par1[2][0] = -0.332171;
  par2[2][0] = -31.8456;
  par3[2][0] = 22.543;

  par0[2][1] = 0.0109109;
  par1[2][1] = 0.0425903;
  par2[2][1] = 6.52561;
  par3[2][1] = 2.18593e-08;

  par0[2][2] = 0;
  par1[2][2] = 0;
  par2[2][2] = 0;
  par3[2][2] = 0;

  par0[2][3] = 0;
  par1[2][3] = 0;
  par2[2][3] = 0;
  par3[2][3] = 0;

  par0[2][4] = 0;
  par1[2][4] = 0;
  par2[2][4] = 0;
  par3[2][4] = 0;

  par0[2][5] = 0;
  par1[2][5] = 0;
  par2[2][5] = 0;
  par3[2][5] = 0;

  par0[3][0] = 0.00343003;
  par1[3][0] = 11.5791;
  par2[3][0] = -112.084;
  par3[3][0] = -863.968;

  par0[3][1] = 0.0372159;
  par1[3][1] = 1.44028;
  par2[3][1] = -40;
  par3[3][1] = 0.00102639;

  par0[3][2] = 0;
  par1[3][2] = 0;
  par2[3][2] = 0;
  par3[3][2] = 0;

  par0[3][3] = 0;
  par1[3][3] = 0;
  par2[3][3] = 0;
  par3[3][3] = 0;

  par0[3][4] = 0;
  par1[3][4] = 0;
  par2[3][4] = 0;
  par3[3][4] = 0;

  par0[3][5] = 0;
  par1[3][5] = 0;
  par2[3][5] = 0;
  par3[3][5] = 0;

  par0[4][0] = 0.0192411;
  par1[4][0] = 0.0511006;
  par2[4][0] = 7.56304;
  par3[4][0] = 0.00331583;

  par0[4][1] = 0.0195124;
  par1[4][1] = 0.104321;
  par2[4][1] = 5.71476;
  par3[4][1] = 6.12472e-06;

  par0[4][2] = 0;
  par1[4][2] = 0;
  par2[4][2] = 0;
  par3[4][2] = 0;

  par0[4][3] = 0;
  par1[4][3] = 0;
  par2[4][3] = 0;
  par3[4][3] = 0;

  par0[4][4] = 0;
  par1[4][4] = 0;
  par2[4][4] = 0;
  par3[4][4] = 0;

  par0[4][5] = 0;
  par1[4][5] = 0;
  par2[4][5] = 0;
  par3[4][5] = 0;

  par0[5][0] = 0.0203644;
  par1[5][0] = -0.050789;
  par2[5][0] = -7.96854;
  par3[5][0] = 4.71223;

  par0[5][1] = 0.0198718;
  par1[5][1] = 0.106859;
  par2[5][1] = 3.54235;
  par3[5][1] = 6.89631e-06;

  par0[5][2] = 0;
  par1[5][2] = 0;
  par2[5][2] = 0;
  par3[5][2] = 0;

  par0[5][3] = 0;
  par1[5][3] = 0;
  par2[5][3] = 0;
  par3[5][3] = 0;

  par0[5][4] = 0;
  par1[5][4] = 0;
  par2[5][4] = 0;
  par3[5][4] = 0;

  par0[5][5] = 0;
  par1[5][5] = 0;
  par2[5][5] = 0;
  par3[5][5] = 0;

  int iEtaSl = -1;
  for (int iEta = 0; iEta < nBinsEta; ++iEta) {
    if (EtaBins[iEta] <= std::abs(eta) && std::abs(eta) < EtaBins[iEta + 1]) {
      iEtaSl = iEta;
    }
  }

  int iBremSl = -1;
  for (int iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
    if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
      iBremSl = iBrem;
    }
  }

  if (std::abs(eta) > 2.5)
    iEtaSl = nBinsEta - 1;
  if (brem < BremBins[0])
    iBremSl = 0;
  if (brem > BremBins[nBinsBrem - 1])
    iBremSl = nBinsBrem - 1;
  if (brem > 2)
    iBremSl = 1;

  float uncertainty = 0;
  if (iBremSl >= 0 && iBremSl < nBinsBrem && iEtaSl >= 0 && iEtaSl < nBinsEta) {
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
    else if (et > 200)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (200 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((200 - par2[iEtaSl][iBremSl]) * (200 - par2[iEtaSl][iBremSl]));
    else if (et >= 5 && et <= 200)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));
  }

  return (uncertainty * energy);
}
