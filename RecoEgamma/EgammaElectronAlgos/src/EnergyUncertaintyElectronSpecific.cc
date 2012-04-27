#include "RecoEgamma/EgammaElectronAlgos/interface/EnergyUncertaintyElectronSpecific.h"
#include "TMath.h"

#include <iostream>


//EnergyUncertaintyElectronSpecific::EnergyUncertaintyElectronSpecific(  const edm::ParameterSet& config ) {
//}

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
  if (c == reco::GsfElectron::GAP) return computeElectronEnergyUncertainty_crack(eta,brem,energy) ;
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

  par0[0][0]=0.00569485;
  par1[0][0]=0.238031;
  par2[0][0]=2.14418;

  par0[0][1]=0.00654717;
  par1[0][1]=0.195018;
  par2[0][1]=3.36553;

  par0[0][2]=0.00574686;
  par1[0][2]=0.24921;
  par2[0][2]=1.74038;

  par0[0][3]=0.00542005;
  par1[0][3]=0.260247;
  par2[0][3]=1.4629;

  par0[0][4]=0.00519581;
  par1[0][4]=0.313224;
  par2[0][4]=0.128454;

  par0[0][5]=0.00548378;
  par1[0][5]=0.389885;
  par2[0][5]=-2.75898;

  par0[1][0]=0.00555461;
  par1[1][0]=0.287409;
  par2[1][0]=1.3559;

  par0[1][1]=0.00610875;
  par1[1][1]=0.312557;
  par2[1][1]=0.136187;

  par0[1][2]=0.00626212;
  par1[1][2]=0.295526;
  par2[1][2]=0.63609;

  par0[1][3]=0.00574565;
  par1[1][3]=0.295161;
  par2[1][3]=0.76533;

  par0[1][4]=0.0044733;
  par1[1][4]=0.379547;
  par2[1][4]=-1.43815;

  par0[1][5]=0.00597035;
  par1[1][5]=0.380462;
  par2[1][5]=-2.30301;

  par0[2][0]=0.00356971;
  par1[2][0]=0.467795;
  par2[2][0]=0.589597;

  par0[2][1]=0.00522785;
  par1[2][1]=0.393588;
  par2[2][1]=1.01088;

  par0[2][2]=0.00366537;
  par1[2][2]=0.525694;
  par2[2][2]=-1.31264;

  par0[2][3]=0.00597205;
  par1[2][3]=0.435691;
  par2[2][3]=0.701969;

  par0[2][4]=0.0048076;
  par1[2][4]=0.53886;
  par2[2][4]=-0.994036;

  par0[2][5]=0.0052884;
  par1[2][5]=0.723095;
  par2[2][5]=-4.43475;

  par0[3][0]=0.0109913;
  par1[3][0]=1.0591;
  par2[3][0]=-2.99534;

  par0[3][1]=0.00873714;
  par1[3][1]=1.41897;
  par2[3][1]=-7.27613;

  par0[3][2]=0.0105263;
  par1[3][2]=1.40034;
  par2[3][2]=-7.42131;

  par0[3][3]=0.0111235;
  par1[3][3]=1.40918;
  par2[3][3]=-7.32411;

  par0[3][4]=0.0154815;
  par1[3][4]=1.38168;
  par2[3][4]=-7.72741;

  par0[3][5]=0.00789044;
  par1[3][5]=2.65891;
  par2[3][5]=-20.6616;

  par0[4][0]=0.011218;
  par1[4][0]=0.441645;
  par2[4][0]=1.09207;

  par0[4][1]=0.0108391;
  par1[4][1]=0.576368;
  par2[4][1]=-1.73995;

  par0[4][2]=0.0102145;
  par1[4][2]=0.673293;
  par2[4][2]=-2.59657;

  par0[4][3]=0.00922011;
  par1[4][3]=0.893962;
  par2[4][3]=-6.07598;

  par0[4][4]=0.010131;
  par1[4][4]=1.14249;
  par2[4][4]=-9.65473;

  par0[4][5]=0.00903571;
  par1[4][5]=1.87983;
  par2[4][5]=-18.1804;


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

  par0[0][0]=0.00313854;
  par1[0][0]=0.508819;
  par2[0][0]=0.842228;

  par0[1][0]=0.00288374;
  par1[1][0]=0.85921;
  par2[1][0]=-1.04733;

  par0[2][0]=-0.00304436;
  par1[2][0]=3.06264;
  par2[2][0]=-17.122;

  par0[3][0]=0.00738289;
  par1[3][0]=1.14414;
  par2[3][0]=-5.03161;


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
double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_badtrack(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.7, 1.3, 1.8, 2.5};

  const int nBinsBrem=1;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];

  par0[0][0]=0.00515087;
  par1[0][0]=0.485648;
  par2[0][0]=0.469686;

  par0[1][0]=0.00583982;
  par1[1][0]=1.28274;
  par2[1][0]=-4.94407;

  par0[2][0]=0.000125585;
  par1[2][0]=2.73223;
  par2[2][0]=-12.2702;

  par0[3][0]=-0.0021395;
  par1[3][0]=2.7229;
  par2[3][0]=-16.1032;


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

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_showering(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0,  0.8,  1.2,  1.7,  2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  1.8,  2.2,  3.0,  4.0,  8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];

  par0[0][0]=0.00521281;
  par1[0][0]=0.559457;
  par2[0][0]=-6.3168;

  par0[0][1]=0.0057348;
  par1[0][1]=0.49441;
  par2[0][1]=-3.3697;

  par0[0][2]=0.00494689;
  par1[0][2]=0.569071;
  par2[0][2]=-4.28264;

  par0[0][3]=0.00471525;
  par1[0][3]=0.631106;
  par2[0][3]=-4.33038;

  par0[0][4]=0.00441388;
  par1[0][4]=0.689909;
  par2[0][4]=-2.69986;

  par0[1][0]=0.00218929;
  par1[1][0]=1.01839;
  par2[1][0]=-8.16055;

  par0[1][1]=0.00624807;
  par1[1][1]=0.83824;
  par2[1][1]=-4.87593;

  par0[1][2]=0.00686613;
  par1[1][2]=0.993807;
  par2[1][2]=-6.67775;

  par0[1][3]=0.0111092;
  par1[1][3]=0.923758;
  par2[1][3]=-4.53745;

  par0[1][4]=0.0129711;
  par1[1][4]=1.26015;
  par2[1][4]=-6.88663;

  par0[2][0]=-0.00343001;
  par1[2][0]=2.53025;
  par2[2][0]=-13.8336;

  par0[2][1]=0.0060143;
  par1[2][1]=1.61835;
  par2[2][1]=-6.91398;

  par0[2][2]=0.00269716;
  par1[2][2]=1.97701;
  par2[2][2]=-9.48542;

  par0[2][3]=0.00165667;
  par1[2][3]=2.20775;
  par2[2][3]=-11.8617;

  par0[2][4]=0.00496115;
  par1[2][4]=1.96708;
  par2[2][4]=-9.46142;

  par0[3][0]=0.0167539;
  par1[3][0]=0.896567;
  par2[3][0]=-0.85402;

  par0[3][1]=0.0176164;
  par1[3][1]=1.5133;
  par2[3][1]=-8.90135;

  par0[3][2]=0.0151386;
  par1[3][2]=2.21006;
  par2[3][2]=-14.7005;

  par0[3][3]=0.000782401;
  par1[3][3]=4.23641;
  par2[3][3]=-28.7204;

  par0[3][4]=-0.000277105;
  par1[3][4]=3.85319;
  par2[3][4]=-21.4914;


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

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_crack(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=4;
  const Double_t EtaBins[nBinsEta+1]  = {0.0,  0.7,  1.56, 1.65};

  const int nBinsBrem=2;
  const Double_t BremBins[nBinsBrem+1]= {0.8,  2, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];

  par0[0][0]=0.00780418;
  par1[0][0]=0.821402;
  par2[0][0]=-7.28348;

  par0[0][1]=0.00508759;
  par1[0][1]=1.12372;
  par2[0][1]=-9.54067;

  par0[1][0]=-0.00721075;
  par1[1][0]=3.26748;
  par2[1][0]=-35.0937;

  par0[1][1]=0.00316556;
  par1[1][1]=2.38557;
  par2[1][1]=-15.2891;

  par0[2][0]=-0.0403143;
  par1[2][0]=9.99996;
  par2[2][0]=-20.5077;

  par0[2][1]=-0.0361807;
  par1[2][1]=9.99999;
  par2[2][1]=-34.7196;

  par0[3][0]=-0.00700281;
  par1[3][0]=2.90173;
  par2[3][0]=-21.0255;

  par0[3][1]=0.000467181;
  par1[3][1]=3.07911;
  par2[3][1]=-14.7479;


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
