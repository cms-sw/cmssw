#include "RecoEgamma/EgammaPhotonAlgos/interface/EnergyUncertaintyPhotonSpecific.h"

#include <iostream>

EnergyUncertaintyPhotonSpecific::EnergyUncertaintyPhotonSpecific(const edm::ParameterSet& config) {}

EnergyUncertaintyPhotonSpecific::~EnergyUncertaintyPhotonSpecific() {}

void EnergyUncertaintyPhotonSpecific::init(const edm::EventSetup& theEventSetup) {}

double EnergyUncertaintyPhotonSpecific::computePhotonEnergyUncertainty_lowR9(double eta, double brem, double energy) {
  double et = energy / cosh(eta);

  constexpr int nBinsEta = 6;
  const double EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  constexpr int nBinsBrem = 6;
  const double BremBins[nBinsBrem + 1] = {0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par0 = {
      {{{0.0232291, 0.00703187, 0.00692465, 0.00855993, 0.00795058, 0.0107494}},
       {{0.0614866, 0.00894211, 0.0102959, 0.0128934, 0.0130199, 0.0180839}},
       {{0.0291343, 0.00876269, 0.0120863, 0.0112655, 0.0168267, 0.0168059}},
       {{0.158403, 0.0717431, 0.0385666, 0.0142631, 0.0421638, 0.046331}},
       {{0.0483944, 0.0168516, 0.0243039, 0.031795, 0.0414953, 0.058031}},
       {{0.107158, 0.021685, 0.0196619, 0.0324734, 0.0414953, 0.058031}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par1 = {
      {{{0., 0.646644, 0.292698, 0.280843, 0.370007, 0.276159}},
       {{0., 0.466937, 0.313568, 0.302943, 0.505135, 0.382134}},
       {{0., 0.375159, 0.397635, 0.856565, 0.636468, 1.09268}},
       {{0., 1.66981, 3.6319, 8.85991, 3.1289, 1.29951}},
       {{0., 1.19617, 0.994626, 0.875925, 0.654605, 0.292915}},
       {{0., 0.574207, 0.940217, 0.574766, 0.654605, 0.292915}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par2 = {
      {{{0., -7.4698, 4.16907, 4.25527, 3.03429, 4.44532}},
       {{0., 3.33434, 6.34301, 6.35598, 2.52964, 5.3388}},
       {{0., 7.11411, 5.97451, -5.76122, -1.54548, -0.547554}},
       {{0., 6.86275, -3.76633, -32.6073, -6.58653, 1.76117}},
       {{0., -6.78666, -4.26073, 1.43183, 4.45367, 8.48307}},
       {{0., -0.566981, -6.05845, -5.23571, 4.45367, 8.48307}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par3 = {
      {{{0., 5.53373e-08, 5.61149e-06, 9.6404e-07, 4.43986e-07, 2.58822e-06}},
       {{0., 0.000114835, 2.86726e-07, 0.00190694, 0.120204, 3.59921e-07}},
       {{0., 0.0438575, 0.0469782, 4.99993, 4.99992, 0.0952985}},
       {{0., 0.00543544, 6.56718e-05, 0.00119538, 1.10125e-05, 0.00204206}},
       {{0., 4.98192, 4.99984, 0.0920944, 0.030385, 0.0134321}},
       {{0., 0.0120609, 0.000193818, 4.9419, 0.030385, 0.0134321}}}};

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
  if (iBremSl >= 0 && iBremSl < nBinsBrem && iEtaSl >= 0 && iEtaSl < nBinsEta) {
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
    if (et > 200)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (200 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((200 - par2[iEtaSl][iBremSl]) * (200 - par2[iEtaSl][iBremSl]));

    if (et > 5 && et < 200)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));
  }

  return (uncertainty * energy);
}

double EnergyUncertaintyPhotonSpecific::computePhotonEnergyUncertainty_highR9(double eta, double brem, double energy) {
  double et = energy / cosh(eta);

  constexpr int nBinsEta = 6;
  const double EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  constexpr int nBinsBrem = 2;
  const double BremBins[nBinsBrem + 1] = {0.8, 1.0, 2.0};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par0 = {{{{0.00806753, 0.00899298}},
                                                                        {{0.00880649, 0.00972275}},
                                                                        {{0.0101474, 0.0109109}},
                                                                        {{0.00343003, 0.0372159}},
                                                                        {{0.0192411, 0.0195124}},
                                                                        {{0.0203644, 0.0198718}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par1 = {{{{0.143754, 0.10159}},
                                                                        {{0.0716169, 0.0752675}},
                                                                        {{-0.332171, 0.0425903}},
                                                                        {{11.5791, 1.44028}},
                                                                        {{0.0511006, 0.104321}},
                                                                        {{-0.050789, 0.106859}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par2 = {{{{-0.00368104, 4.70884}},
                                                                        {{5.23856, 3.35623}},
                                                                        {{-31.8456, 6.52561}},
                                                                        {{-112.084, -40.}},
                                                                        {{7.56304, 5.71476}},
                                                                        {{-7.96854, 3.54235}}}};

  constexpr std::array<std::array<float, nBinsBrem>, nBinsEta> par3 = {{{{0.219829, 9.07419e-08}},
                                                                        {{0.00632907, 2.49397e-07}},
                                                                        {{22.543, 2.18593e-08}},
                                                                        {{-863.968, 0.00102639}},
                                                                        {{0.00331583, 6.12472e-06}},
                                                                        {{4.71223, 6.89631e-06}}}};

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
