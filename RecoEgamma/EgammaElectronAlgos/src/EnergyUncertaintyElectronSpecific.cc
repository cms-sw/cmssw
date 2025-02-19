#include "RecoEgamma/EgammaElectronAlgos/interface/EnergyUncertaintyElectronSpecific.h"
#include "TMath.h"

#include <iostream>


EnergyUncertaintyElectronSpecific::EnergyUncertaintyElectronSpecific() {
}

EnergyUncertaintyElectronSpecific::~EnergyUncertaintyElectronSpecific()
 {}

void EnergyUncertaintyElectronSpecific::init(  const edm::EventSetup& theEventSetup )
 {}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty( reco::GsfElectron::Classification c, double eta, double brem, double energy)
 {
  if (c == reco::GsfElectron::GOLDEN) return computeElectronEnergyUncertainty_golden(eta,brem,energy) ;
  if (c == reco::GsfElectron::BIGBREM) return computeElectronEnergyUncertainty_bigbrem(eta,brem,energy) ;
  if (c == reco::GsfElectron::SHOWERING) return computeElectronEnergyUncertainty_showering(eta,brem,energy) ;
  if (c == reco::GsfElectron::BADTRACK) return computeElectronEnergyUncertainty_badtrack(eta,brem,energy) ;
  if (c == reco::GsfElectron::GAP) return computeElectronEnergyUncertainty_cracks(eta,brem,energy) ;
  throw cms::Exception("GsfElectronAlgo|InternalError")<<"unknown classification" ;
 }

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_golden(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=5;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.4, 0.8, 1.5, 2.0, 2.5};

  const int nBinsBrem=6;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  1.0, 1.1, 1.2, 1.3, 1.5, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];

  par0[0][0]=0.00567891;
  par1[0][0]=0.238685;
  par2[0][0]=2.12035;

  par0[0][1]=0.0065673;
  par1[0][1]=0.193642;
  par2[0][1]=3.41493;

  par0[0][2]=0.00574742;
  par1[0][2]=0.249171;
  par2[0][2]=1.7401;

  par0[0][3]=0.00542964;
  par1[0][3]=0.259997;
  par2[0][3]=1.46234;

  par0[0][4]=0.00523293;
  par1[0][4]=0.310505;
  par2[0][4]=0.233226;

  par0[0][5]=0.00547518;
  par1[0][5]=0.390506;
  par2[0][5]=-2.78168;

  par0[1][0]=0.00552517;
  par1[1][0]=0.288736;
  par2[1][0]=1.30552;

  par0[1][1]=0.00611188;
  par1[1][1]=0.312303;
  par2[1][1]=0.137905;

  par0[1][2]=0.0062729;
  par1[1][2]=0.294717;
  par2[1][2]=0.653793;

  par0[1][3]=0.00574846;
  par1[1][3]=0.294491;
  par2[1][3]=0.790746;

  par0[1][4]=0.00447373;
  par1[1][4]=0.379178;
  par2[1][4]=-1.42584;

  par0[1][5]=0.00595789;
  par1[1][5]=0.38164;
  par2[1][5]=-2.34653;

  par0[2][0]=0.00356679;
  par1[2][0]=0.456456;
  par2[2][0]=0.610716;

  par0[2][1]=0.00503827;
  par1[2][1]=0.394912;
  par2[2][1]=0.778879;

  par0[2][2]=0.00328016;
  par1[2][2]=0.541713;
  par2[2][2]=-1.58577;

  par0[2][3]=0.00592303;
  par1[2][3]=0.401744;
  par2[2][3]=1.45098;

  par0[2][4]=0.00512479;
  par1[2][4]=0.483151;
  par2[2][4]=-0.0985911;

  par0[2][5]=0.00484166;
  par1[2][5]=0.657995;
  par2[2][5]=-3.47167;

  par0[3][0]=0.0109195;
  par1[3][0]=1.13803;
  par2[3][0]=-3.48281;

  par0[3][1]=0.0102361;
  par1[3][1]=1.39866;
  par2[3][1]=-6.4736;

  par0[3][2]=0.0101576;
  par1[3][2]=1.51353;
  par2[3][2]=-8.03308;

  par0[3][3]=0.0120683;
  par1[3][3]=1.48587;
  par2[3][3]=-7.55974;

  par0[3][4]=0.0155326;
  par1[3][4]=1.49732;
  par2[3][4]=-7.98843;

  par0[3][5]=0.0225035;
  par1[3][5]=1.82363;
  par2[3][5]=-10.1027;

  par0[4][0]=0.0109632;
  par1[4][0]=0.458212;
  par2[4][0]=0.995183;

  par0[4][1]=0.0103342;
  par1[4][1]=0.628761;
  par2[4][1]=-2.42889;

  par0[4][2]=0.0103486;
  par1[4][2]=0.659144;
  par2[4][2]=-2.14073;

  par0[4][3]=0.00862762;
  par1[4][3]=0.929563;
  par2[4][3]=-6.27768;

  par0[4][4]=0.0111448;
  par1[4][4]=1.06724;
  par2[4][4]=-7.68512;

  par0[4][5]=0.0146648;
  par1[4][5]=1.6427;
  par2[4][5]=-13.3504;


  Int_t iEtaSl = -1;
  for (Int_t iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= fabs(eta) && fabs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  Int_t iBremSl = -1;
  for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem){
    if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (fabs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]);
  if (et>100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(100-par2[iEtaSl][iBremSl]);

  if (et>5 && et<100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]);

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_bigbrem(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0,  0.8,  1.5,  2.0,  2.5};

  const int nBinsBrem=1;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00593389;
  par1[0][0]=0.178275;
  par2[0][0]=-7.28273;
  par3[0][0]=13.2632;

  par0[1][0]=0.00266954;
  par1[1][0]=0.811415;
  par2[1][0]=-1.66063;
  par3[1][0]=1.03555;

  par0[2][0]=0.00500623;
  par1[2][0]=2.34018;
  par2[2][0]=-11.0129;
  par3[2][0]=-0.200323;

  par0[3][0]=0.00841038;
  par1[3][0]=1.06851;
  par2[3][0]=-4.1259;
  par3[3][0]=-0.0646195;


  Int_t iEtaSl = -1;
  for (Int_t iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= fabs(eta) && fabs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  Int_t iBremSl = -1;
  for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem){
    if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (fabs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((5-par2[iEtaSl][iBremSl])*(5-par2[iEtaSl][iBremSl]));
  if (et>100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(100-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((100-par2[iEtaSl][iBremSl])*(100-par2[iEtaSl][iBremSl]));

  if (et>5 && et<100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}
double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_badtrack(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.7, 1.3, 1.8, 2.5};

  const int nBinsBrem=1;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00601311;
  par1[0][0]=0.390988;
  par2[0][0]=-4.11919;
  par3[0][0]=4.61671;

  par0[1][0]=0.0059814;
  par1[1][0]=1.02668;
  par2[1][0]=-2.87477;
  par3[1][0]=0.163447;

  par0[2][0]=0.00953032;
  par1[2][0]=2.27491;
  par2[2][0]=-7.61675;
  par3[2][0]=-0.335786;

  par0[3][0]=0.00728618;
  par1[3][0]=2.08268;
  par2[3][0]=-8.66756;
  par3[3][0]=-1.27831;


  Int_t iEtaSl = -1;
  for (Int_t iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= fabs(eta) && fabs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  Int_t iBremSl = -1;
  for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem){
    if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (fabs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((5-par2[iEtaSl][iBremSl])*(5-par2[iEtaSl][iBremSl]));
  if (et>100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(100-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((100-par2[iEtaSl][iBremSl])*(100-par2[iEtaSl][iBremSl]));

  if (et>5 && et<100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_showering(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0,  0.8,  1.2,  1.7,  2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  1.8,  2.2,  3.0,  4.0,  8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.0049351;
  par1[0][0]=0.579925;
  par2[0][0]=-9.33987;
  par3[0][0]=1.62129;

  par0[0][1]=0.00566155;
  par1[0][1]=0.496137;
  par2[0][1]=-5.52543;
  par3[0][1]=1.19101;

  par0[0][2]=0.0051397;
  par1[0][2]=0.551947;
  par2[0][2]=-7.30079;
  par3[0][2]=1.89701;

  par0[0][3]=0.00468481;
  par1[0][3]=0.63011;
  par2[0][3]=-6.7722;
  par3[0][3]=1.81614;

  par0[0][4]=0.00444475;
  par1[0][4]=0.684261;
  par2[0][4]=-4.67614;
  par3[0][4]=1.64415;

  par0[1][0]=0.00201762;
  par1[1][0]=0.914762;
  par2[1][0]=-4.48042;
  par3[1][0]=-1.50473;

  par0[1][1]=0.00431475;
  par1[1][1]=0.824483;
  par2[1][1]=-5.02885;
  par3[1][1]=-0.153502;

  par0[1][2]=0.00501004;
  par1[1][2]=0.888521;
  par2[1][2]=-4.77311;
  par3[1][2]=-0.355145;

  par0[1][3]=0.00632666;
  par1[1][3]=0.960241;
  par2[1][3]=-3.36742;
  par3[1][3]=-1.16499;

  par0[1][4]=0.00636704;
  par1[1][4]=1.25728;
  par2[1][4]=-5.53561;
  par3[1][4]=-0.864123;

  par0[2][0]=-0.00729396;
  par1[2][0]=3.24295;
  par2[2][0]=-17.1458;
  par3[2][0]=-4.69711;

  par0[2][1]=0.00539783;
  par1[2][1]=1.72935;
  par2[2][1]=-5.92807;
  par3[2][1]=-2.18733;

  par0[2][2]=0.00608149;
  par1[2][2]=1.80606;
  par2[2][2]=-6.67563;
  par3[2][2]=-0.922401;

  par0[2][3]=0.00465335;
  par1[2][3]=2.13562;
  par2[2][3]=-10.1105;
  par3[2][3]=-0.230781;

  par0[2][4]=0.00642685;
  par1[2][4]=2.07592;
  par2[2][4]=-7.50257;
  par3[2][4]=-2.91515;

  par0[3][0]=0.0149449;
  par1[3][0]=1.00448;
  par2[3][0]=-2.09368;
  par3[3][0]=0.455037;

  par0[3][1]=0.0216691;
  par1[3][1]=1.18393;
  par2[3][1]=-4.56674;
  par3[3][1]=-0.601872;

  par0[3][2]=0.0255957;
  par1[3][2]=0.00775295;
  par2[3][2]=-44.2722;
  par3[3][2]=241.516;

  par0[3][3]=0.0206101;
  par1[3][3]=2.59246;
  par2[3][3]=-13.1702;
  par3[3][3]=-2.35024;

  par0[3][4]=0.0180508;
  par1[3][4]=3.1099;
  par2[3][4]=-13.6208;
  par3[3][4]=-2.11069;


  Int_t iEtaSl = -1;
  for (Int_t iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= fabs(eta) && fabs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  Int_t iBremSl = -1;
  for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem){
    if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (fabs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((5-par2[iEtaSl][iBremSl])*(5-par2[iEtaSl][iBremSl]));
  if (et>100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(100-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((100-par2[iEtaSl][iBremSl])*(100-par2[iEtaSl][iBremSl]));

  if (et>5 && et<100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_cracks(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=5;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.42, 0.78, 1.2, 1.52, 1.65};

  const int nBinsBrem=6;
  const Double_t BremBins[nBinsBrem+1]= {0.8, 1.2, 1.5, 2.1, 3., 4, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];

  par0[0][0]=0.0139815;
  par1[0][0]=0.569273;
  par2[0][0]=-4.31243;

  par0[0][1]=0.00550839;
  par1[0][1]=0.674654;
  par2[0][1]=-3.071;

  par0[0][2]=0.0108292;
  par1[0][2]=0.523128;
  par2[0][2]=-2.56702;

  par0[0][3]=0.00596201;
  par1[0][3]=1.02501;
  par2[0][3]=-7.74555;

  par0[0][4]=-0.00498136;
  par1[0][4]=1.75645;
  par2[0][4]=-21.3726;

  par0[0][5]=0.000621696;
  par1[0][5]=0.955191;
  par2[0][5]=-6.2189;

  par0[1][0]=0.00467498;
  par1[1][0]=0.697951;
  par2[1][0]=-6.56009;

  par0[1][1]=0.00808463;
  par1[1][1]=0.580628;
  par2[1][1]=-3.66067;

  par0[1][2]=0.00546665;
  par1[1][2]=0.814515;
  par2[1][2]=-7.8275;

  par0[1][3]=0.00506318;
  par1[1][3]=0.819975;
  par2[1][3]=-6.01641;

  par0[1][4]=0.00608425;
  par1[1][4]=0.829616;
  par2[1][4]=-7.85456;

  par0[1][5]=-4.45641e-06;
  par1[1][5]=1.18952;
  par2[1][5]=-8.27071;

  par0[2][0]=0.00971734;
  par1[2][0]=3.79446;
  par2[2][0]=-49.9996;

  par0[2][1]=0.00063951;
  par1[2][1]=2.47472;
  par2[2][1]=-25.0724;

  par0[2][2]=-0.0121618;
  par1[2][2]=5.12931;
  par2[2][2]=-49.985;

  par0[2][3]=-0.00604365;
  par1[2][3]=3.42497;
  par2[2][3]=-28.1932;

  par0[2][4]=0.00492161;
  par1[2][4]=1.84123;
  par2[2][4]=-10.6485;

  par0[2][5]=-0.00143907;
  par1[2][5]=2.3773;
  par2[2][5]=-15.4014;

  par0[3][0]=-0.0844907;
  par1[3][0]=19.9999;
  par2[3][0]=-39.9444;

  par0[3][1]=-0.0592498;
  par1[3][1]=10.4079;
  par2[3][1]=-25.1133;

  par0[3][2]=-0.0828631;
  par1[3][2]=16.6273;
  par2[3][2]=-49.9999;

  par0[3][3]=-0.0740798;
  par1[3][3]=15.9316;
  par2[3][3]=-50;

  par0[3][4]=-0.0698045;
  par1[3][4]=15.4883;
  par2[3][4]=-49.9998;

  par0[3][5]=-0.0699518;
  par1[3][5]=14.7306;
  par2[3][5]=-49.9998;

  par0[4][0]=-0.0999971;
  par1[4][0]=15.9122;
  par2[4][0]=-30.1268;

  par0[4][1]=-0.0999996;
  par1[4][1]=18.5882;
  par2[4][1]=-42.6113;

  par0[4][2]=-0.0989356;
  par1[4][2]=19.9996;
  par2[4][2]=-46.6999;

  par0[4][3]=-0.0999965;
  par1[4][3]=19.9999;
  par2[4][3]=-47.074;

  par0[4][4]=-0.0833049;
  par1[4][4]=18.2281;
  par2[4][4]=-49.9995;

  par0[4][5]=-0.020072;
  par1[4][5]=8.1587;
  par2[4][5]=-25.2897;


  Int_t iEtaSl = -1;
  for (Int_t iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= fabs(eta) && fabs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  Int_t iBremSl = -1;
  for (Int_t iBrem = 0; iBrem < nBinsBrem; ++iBrem){
    if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (fabs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]);
  if (et>100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(100-par2[iEtaSl][iBremSl]);

  if (et>5 && et<100) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]);

  return (uncertainty*energy);

}
