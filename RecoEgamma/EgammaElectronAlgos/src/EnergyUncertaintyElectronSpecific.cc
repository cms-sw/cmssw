#include "RecoEgamma/EgammaElectronAlgos/interface/EnergyUncertaintyElectronSpecific.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {

  double computeEnergyUncertaintyGolden(double eta, double brem, double energy) {
    double et = energy / cosh(eta);

    constexpr int nBinsEta = 5;
    constexpr double EtaBins[nBinsEta + 1] = {0.0, 0.4, 0.8, 1.5, 2.0, 2.5};

    constexpr int nBinsBrem = 6;
    constexpr double BremBins[nBinsBrem + 1] = {0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 8.0};

    constexpr float par0[nBinsEta][nBinsBrem] = {
        {0.00567891, 0.0065673, 0.00574742, 0.00542964, 0.00523293, 0.00547518},

        {0.00552517, 0.00611188, 0.0062729, 0.00574846, 0.00447373, 0.00595789},

        {0.00356679, 0.00503827, 0.00328016, 0.00592303, 0.00512479, 0.00484166},

        {0.0109195, 0.0102361, 0.0101576, 0.0120683, 0.0155326, 0.0225035},

        {0.0109632, 0.0103342, 0.0103486, 0.00862762, 0.0111448, 0.0146648}};

    static_assert(par0[0][0] == 0.00567891f);
    static_assert(par0[0][1] == 0.0065673f);
    static_assert(par0[1][3] == 0.00574846f);

    constexpr float par1[nBinsEta][nBinsBrem] = {{0.238685, 0.193642, 0.249171, 0.259997, 0.310505, 0.390506},

                                                 {0.288736, 0.312303, 0.294717, 0.294491, 0.379178, 0.38164},

                                                 {0.456456, 0.394912, 0.541713, 0.401744, 0.483151, 0.657995},

                                                 {1.13803, 1.39866, 1.51353, 1.48587, 1.49732, 1.82363},

                                                 {0.458212, 0.628761, 0.659144, 0.929563, 1.06724, 1.6427}};

    constexpr float par2[nBinsEta][nBinsBrem] = {{2.12035, 3.41493, 1.7401, 1.46234, 0.233226, -2.78168},

                                                 {1.30552, 0.137905, 0.653793, 0.790746, -1.42584, -2.34653},

                                                 {0.610716, 0.778879, -1.58577, 1.45098, -0.0985911, -3.47167},
                                                 {-3.48281, -6.4736, -8.03308, -7.55974, -7.98843, -10.1027},

                                                 {0.995183, -2.42889, -2.14073, -6.27768, -7.68512, -13.3504}};

    Int_t iEtaSl = -1;
    for (Int_t iEta = 0; iEta < nBinsEta; ++iEta) {
      if (EtaBins[iEta] <= fabs(eta) && fabs(eta) < EtaBins[iEta + 1]) {
        iEtaSl = iEta;
      }
    }

    Int_t iBremSl = -1;
    for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
      if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
        iBremSl = iBrem;
      }
    }

    if (fabs(eta) > 2.5)
      iEtaSl = nBinsEta - 1;
    if (brem < BremBins[0])
      iBremSl = 0;
    if (brem > BremBins[nBinsBrem - 1])
      iBremSl = nBinsBrem - 1;

    if (iEtaSl == -1) {
      edm::LogError("BadRange") << "Bad eta value: " << eta << " in computeEnergyUncertaintyGolden";
      return 0;
    }

    if (iBremSl == -1) {
      edm::LogError("BadRange") << "Bad brem value: " << brem << " in computeEnergyUncertaintyGolden";
      return 0;
    }

    float uncertainty = 0;
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]);
    if (et > 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (100 - par2[iEtaSl][iBremSl]);

    if (et > 5 && et < 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]);

    return (uncertainty * energy);
  }

  double computeEnergyUncertaintyBigbrem(double eta, double brem, double energy) {
    const double et = energy / cosh(eta);

    constexpr int nBinsEta = 4;
    constexpr double EtaBins[nBinsEta + 1] = {0.0, 0.8, 1.5, 2.0, 2.5};

    constexpr int nBinsBrem = 1;
    constexpr double BremBins[nBinsBrem + 1] = {0.8, 8.0};

    constexpr float par0[nBinsEta][nBinsBrem] = {{0.00593389}, {0.00266954}, {0.00500623}, {0.00841038}};

    constexpr float par1[nBinsEta][nBinsBrem] = {{0.178275}, {0.811415}, {2.34018}, {1.06851}};

    constexpr float par2[nBinsEta][nBinsBrem] = {{-7.28273}, {-1.66063}, {-11.0129}, {-4.1259}};

    constexpr float par3[nBinsEta][nBinsBrem] = {{13.2632}, {1.03555}, {-0.200323}, {-0.0646195}};

    Int_t iEtaSl = -1;
    for (Int_t iEta = 0; iEta < nBinsEta; ++iEta) {
      if (EtaBins[iEta] <= fabs(eta) && fabs(eta) < EtaBins[iEta + 1]) {
        iEtaSl = iEta;
      }
    }

    Int_t iBremSl = -1;
    for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
      if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
        iBremSl = iBrem;
      }
    }

    if (fabs(eta) > 2.5)
      iEtaSl = nBinsEta - 1;
    if (brem < BremBins[0])
      iBremSl = 0;
    if (brem > BremBins[nBinsBrem - 1])
      iBremSl = nBinsBrem - 1;

    if (iEtaSl == -1) {
      edm::LogError("BadRange") << "Bad eta value: " << eta << " in computeEnergyUncertaintyBigbrem";
      return 0;
    }

    if (iBremSl == -1) {
      edm::LogError("BadRange") << "Bad brem value: " << brem << " in computeEnergyUncertaintyBigbrem";
      return 0;
    }

    float uncertainty = 0;
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
    if (et > 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (100 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((100 - par2[iEtaSl][iBremSl]) * (100 - par2[iEtaSl][iBremSl]));

    if (et > 5 && et < 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));

    return (uncertainty * energy);
  }
  double computeEnergyUncertaintyBadTrack(double eta, double brem, double energy) {
    const double et = energy / cosh(eta);

    constexpr int nBinsEta = 4;
    constexpr double EtaBins[nBinsEta + 1] = {0.0, 0.7, 1.3, 1.8, 2.5};

    constexpr int nBinsBrem = 1;
    constexpr double BremBins[nBinsBrem + 1] = {0.8, 8.0};

    constexpr float par0[nBinsEta][nBinsBrem] = {{0.00601311}, {0.0059814}, {0.00953032}, {0.00728618}};

    constexpr float par1[nBinsEta][nBinsBrem] = {{0.390988}, {1.02668}, {2.27491}, {2.08268}};

    constexpr float par2[nBinsEta][nBinsBrem] = {{-4.11919}, {-2.87477}, {-7.61675}, {-8.66756}};

    constexpr float par3[nBinsEta][nBinsBrem] = {{4.61671}, {0.163447}, {-0.335786}, {-1.27831}};

    Int_t iEtaSl = -1;
    for (Int_t iEta = 0; iEta < nBinsEta; ++iEta) {
      if (EtaBins[iEta] <= fabs(eta) && fabs(eta) < EtaBins[iEta + 1]) {
        iEtaSl = iEta;
      }
    }

    Int_t iBremSl = -1;
    for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
      if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
        iBremSl = iBrem;
      }
    }

    if (fabs(eta) > 2.5)
      iEtaSl = nBinsEta - 1;
    if (brem < BremBins[0])
      iBremSl = 0;
    if (brem > BremBins[nBinsBrem - 1])
      iBremSl = nBinsBrem - 1;

    if (iEtaSl == -1) {
      edm::LogError("BadRange") << "Bad eta value: " << eta << " in computeEnergyUncertaintyBadTrack";
      return 0;
    }

    if (iBremSl == -1) {
      edm::LogError("BadRange") << "Bad brem value: " << brem << " in computeEnergyUncertaintyBadTrack";
      return 0;
    }

    float uncertainty = 0;
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
    if (et > 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (100 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((100 - par2[iEtaSl][iBremSl]) * (100 - par2[iEtaSl][iBremSl]));

    if (et > 5 && et < 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));

    return (uncertainty * energy);
  }

  double computeEnergyUncertaintyShowering(double eta, double brem, double energy) {
    const double et = energy / cosh(eta);

    constexpr int nBinsEta = 4;
    constexpr double EtaBins[nBinsEta + 1] = {0.0, 0.8, 1.2, 1.7, 2.5};

    constexpr int nBinsBrem = 5;
    constexpr double BremBins[nBinsBrem + 1] = {0.8, 1.8, 2.2, 3.0, 4.0, 8.0};

    constexpr float par0[nBinsEta][nBinsBrem] = {{0.0049351, 0.00566155, 0.0051397, 0.00468481, 0.00444475},

                                                 {0.00201762, 0.00431475, 0.00501004, 0.00632666, 0.00636704},

                                                 {-0.00729396, 0.00539783, 0.00608149, 0.00465335, 0.00642685},

                                                 {0.0149449, 0.0216691, 0.0255957, 0.0206101, 0.0180508}};

    constexpr float par1[nBinsEta][nBinsBrem] = {{0.579925, 0.496137, 0.551947, 0.63011, 0.684261},

                                                 {0.914762, 0.824483, 0.888521, 0.960241, 1.25728},

                                                 {3.24295, 1.72935, 1.80606, 2.13562, 2.07592},

                                                 {1.00448, 1.18393, 0.00775295, 2.59246, 3.1099}};

    constexpr float par2[nBinsEta][nBinsBrem] = {{-9.33987, -5.52543, -7.30079, -6.7722, -4.67614},

                                                 {-4.48042, -5.02885, -4.77311, -3.36742, -5.53561},

                                                 {-17.1458, -5.92807, -6.67563, -10.1105, -7.50257},

                                                 {-2.09368, -4.56674, -44.2722, -13.1702, -13.6208}};

    constexpr float par3[nBinsEta][nBinsBrem] = {{1.62129, 1.19101, 1.89701, 1.81614, 1.64415},

                                                 {-1.50473, -0.153502, -0.355145, -1.16499, -0.864123},

                                                 {-4.69711, -2.18733, -0.922401, -0.230781, -2.91515},

                                                 {0.455037, -0.601872, 241.516, -2.35024, -2.11069}};

    Int_t iEtaSl = -1;
    for (Int_t iEta = 0; iEta < nBinsEta; ++iEta) {
      if (EtaBins[iEta] <= fabs(eta) && fabs(eta) < EtaBins[iEta + 1]) {
        iEtaSl = iEta;
      }
    }

    Int_t iBremSl = -1;
    for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
      if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
        iBremSl = iBrem;
      }
    }

    if (fabs(eta) > 2.5)
      iEtaSl = nBinsEta - 1;
    if (brem < BremBins[0])
      iBremSl = 0;
    if (brem > BremBins[nBinsBrem - 1])
      iBremSl = nBinsBrem - 1;

    if (iEtaSl == -1) {
      edm::LogError("BadRange") << "Bad eta value: " << eta << " in computeEnergyUncertaintyShowering";
      return 0;
    }

    if (iBremSl == -1) {
      edm::LogError("BadRange") << "Bad brem value: " << brem << " in computeEnergyUncertaintyShowering";
      return 0;
    }

    float uncertainty = 0;
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((5 - par2[iEtaSl][iBremSl]) * (5 - par2[iEtaSl][iBremSl]));
    if (et > 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (100 - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((100 - par2[iEtaSl][iBremSl]) * (100 - par2[iEtaSl][iBremSl]));

    if (et > 5 && et < 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]) +
                    par3[iEtaSl][iBremSl] / ((et - par2[iEtaSl][iBremSl]) * (et - par2[iEtaSl][iBremSl]));

    return (uncertainty * energy);
  }

  double computeEnergyUncertaintyCracks(double eta, double brem, double energy) {
    const double et = energy / cosh(eta);

    constexpr int nBinsEta = 5;
    constexpr double EtaBins[nBinsEta + 1] = {0.0, 0.42, 0.78, 1.2, 1.52, 1.65};

    constexpr int nBinsBrem = 6;
    constexpr double BremBins[nBinsBrem + 1] = {0.8, 1.2, 1.5, 2.1, 3., 4, 8.0};

    constexpr float par0[nBinsEta][nBinsBrem] = {
        {0.0139815, 0.00550839, 0.0108292, 0.00596201, -0.00498136, 0.000621696},

        {0.00467498, 0.00808463, 0.00546665, 0.00506318, 0.00608425, -4.45641e-06},

        {0.00971734, 0.00063951, -0.0121618, -0.00604365, 0.00492161, -0.00143907},

        {-0.0844907, -0.0592498, -0.0828631, -0.0740798, -0.0698045, -0.0699518},

        {-0.0999971, -0.0999996, -0.0989356, -0.0999965, -0.0833049, -0.020072}};

    constexpr float par1[nBinsEta][nBinsBrem] = {{0.569273, 0.674654, 0.523128, 1.02501, 1.75645, 0.955191},

                                                 {0.697951, 0.580628, 0.814515, 0.819975, 0.829616, 1.18952},

                                                 {3.79446, 2.47472, 5.12931, 3.42497, 1.84123, 2.3773},

                                                 {19.9999, 10.4079, 16.6273, 15.9316, 15.4883, 14.7306},

                                                 {15.9122, 18.5882, 19.9996, 19.9999, 18.2281, 8.1587}};

    constexpr float par2[nBinsEta][nBinsBrem] = {{-4.31243, -3.071, -2.56702, -7.74555, -21.3726, -6.2189},

                                                 {-6.56009, -3.66067, -7.8275, -6.01641, -7.85456, -8.27071},

                                                 {-49.9996, -25.0724, -49.985, -28.1932, -10.6485, -15.4014},

                                                 {-39.9444, -25.1133, -49.9999, -50, -49.9998, -49.9998},

                                                 {-30.1268, -42.6113, -46.6999, -47.074, -49.9995, -25.2897}};

    static_assert(par0[0][3] == 0.00596201f);
    static_assert(par1[0][3] == 1.02501f);
    static_assert(par2[0][3] == -7.74555f);

    static_assert(par0[2][4] == 0.00492161f);
    static_assert(par1[2][4] == 1.84123f);
    static_assert(par2[2][4] == -10.6485f);

    static_assert(par0[4][3] == -0.0999965f);
    static_assert(par1[4][3] == 19.9999f);
    static_assert(par2[4][3] == -47.074f);

    Int_t iEtaSl = -1;
    for (Int_t iEta = 0; iEta < nBinsEta; ++iEta) {
      if (EtaBins[iEta] <= fabs(eta) && fabs(eta) < EtaBins[iEta + 1]) {
        iEtaSl = iEta;
      }
    }

    Int_t iBremSl = -1;
    for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem) {
      if (BremBins[iBrem] <= brem && brem < BremBins[iBrem + 1]) {
        iBremSl = iBrem;
      }
    }

    if (fabs(eta) > 2.5)
      iEtaSl = nBinsEta - 1;
    if (brem < BremBins[0])
      iBremSl = 0;
    if (brem > BremBins[nBinsBrem - 1])
      iBremSl = nBinsBrem - 1;

    if (iEtaSl == -1) {
      edm::LogError("BadRange") << "Bad eta value: " << eta << " in computeEnergyUncertaintyCracks";
      return 0;
    }

    if (iBremSl == -1) {
      edm::LogError("BadRange") << "Bad brem value: " << brem << " in computeEnergyUncertaintyCracks";
      return 0;
    }

    float uncertainty = 0;
    if (et < 5)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (5 - par2[iEtaSl][iBremSl]);
    if (et > 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (100 - par2[iEtaSl][iBremSl]);

    if (et > 5 && et < 100)
      uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl] / (et - par2[iEtaSl][iBremSl]);

    return (uncertainty * energy);
  }

}  // namespace

using reco::GsfElectron;

double egamma::electronEnergyUncertainty(GsfElectron::Classification c, double eta, double brem, double energy) {
  if (c == GsfElectron::GOLDEN)
    return computeEnergyUncertaintyGolden(eta, brem, energy);
  if (c == GsfElectron::BIGBREM)
    return computeEnergyUncertaintyBigbrem(eta, brem, energy);
  if (c == GsfElectron::SHOWERING)
    return computeEnergyUncertaintyShowering(eta, brem, energy);
  if (c == GsfElectron::BADTRACK)
    return computeEnergyUncertaintyBadTrack(eta, brem, energy);
  if (c == GsfElectron::GAP)
    return computeEnergyUncertaintyCracks(eta, brem, energy);
  throw cms::Exception("GsfElectronAlgo|InternalError") << "unknown classification";
}
