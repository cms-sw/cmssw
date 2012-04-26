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
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.4, 0.8, 1.44, 1.56, 2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8, 1.2, 1.4,  1.6,  2.5, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00683679;
  par1[0][0]=0.169381;
  par2[0][0]=6.66481;
  par3[0][0]=2.15944e-06;

  par0[0][1]=0.00702865;
  par1[0][1]=0.176018;
  par2[0][1]=6.61454;
  par3[0][1]=8.25306e-06;

  par0[0][2]=0.00499921;
  par1[0][2]=0.31181;
  par2[0][2]=2.65352;
  par3[0][2]=3.66313e-06;

  par0[0][3]=0.00804359;
  par1[0][3]=0.252666;
  par2[0][3]=3.10194;
  par3[0][3]=1.84312e-06;

  par0[0][4]=0.00383855;
  par1[0][4]=0.270566;
  par2[0][4]=6.01167;
  par3[0][4]=2.12517e-05;

  par0[1][0]=0.00721818;
  par1[1][0]=0.208125;
  par2[1][0]=5.89528;
  par3[1][0]=7.2793e-06;

  par0[1][1]=0.00661331;
  par1[1][1]=0.22829;
  par2[1][1]=5.17432;
  par3[1][1]=5.97592e-06;

  par0[1][2]=0.00661331;
  par1[1][2]=0.22829;
  par2[1][2]=5.17432;
  par3[1][2]=5.97592e-06;

  par0[1][3]=0.00789405;
  par1[1][3]=0.268577;
  par2[1][3]=2.868;
  par3[1][3]=0.000968289;

  par0[1][4]=0.00244252;
  par1[1][4]=0.34679;
  par2[1][4]=3.82416;
  par3[1][4]=9.29242e-06;

  par0[2][0]=0.00597632;
  par1[2][0]=0.332138;
  par2[2][0]=5.26491;
  par3[2][0]=3.66843e-05;

  par0[2][1]=0.00601032;
  par1[2][1]=0.389665;
  par2[2][1]=4.11783;
  par3[2][1]=1.7331e-05;

  par0[2][2]=0.00650525;
  par1[2][2]=0.422166;
  par2[2][2]=3.17016;
  par3[2][2]=0.000149703;

  par0[2][3]=0.00827296;
  par1[2][3]=0.467401;
  par2[2][3]=2.7484;
  par3[2][3]=6.3139e-07;

  par0[2][4]=-0.0138066;
  par1[2][4]=4.01613;
  par2[2][4]=-39.9999;
  par3[2][4]=0.00049157;

  par0[3][0]=-0.000194559;
  par1[3][0]=1.43941;
  par2[3][0]=2.37888;
  par3[3][0]=6.69158e-05;

  par0[3][1]=0.0138382;
  par1[3][1]=0.756393;
  par2[3][1]=5.74284;
  par3[3][1]=0.000710591;

  par0[3][2]=0.0119722;
  par1[3][2]=0.485413;
  par2[3][2]=2.45197;
  par3[3][2]=4.99482;

  par0[3][3]=0.0119222;
  par1[3][3]=0.691866;
  par2[3][3]=5.93252;
  par3[3][3]=0.023263;

  par0[3][4]=-0.00361756;
  par1[3][4]=1.93889;
  par2[3][4]=-4.3931;
  par3[3][4]=0.108295;

  par0[4][0]=0.0132847;
  par1[4][0]=0.495173;
  par2[4][0]=3.36247;
  par3[4][0]=0.244896;

  par0[4][1]=0.0134688;
  par1[4][1]=0.815579;
  par2[4][1]=-1.20718;
  par3[4][1]=0.612571;

  par0[4][2]=0.0201546;
  par1[4][2]=0.735695;
  par2[4][2]=-3.76293;
  par3[4][2]=4.99972;

  par0[4][3]=0.0275522;
  par1[4][3]=0.757208;
  par2[4][3]=-4.89077;
  par3[4][3]=4.93072;

  par0[4][4]=-0.0652707;
  par1[4][4]=8.71622;
  par2[4][4]=-39.9999;
  par3[4][4]=0.00053324;


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
  if (et>200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(200-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((200-par2[iEtaSl][iBremSl])*(200-par2[iEtaSl][iBremSl]));

  if (et>5 && et<200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_bigbrem(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=5;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.4, 0.8, 1.44, 1.56, 2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8, 1.2, 1.4,  1.6,  2.5, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00171539;
  par1[0][0]=0.585942;
  par2[0][0]=3.9897;
  par3[0][0]=0.000302938;

  par0[0][1]=0.00496019;
  par1[0][1]=0.322198;
  par2[0][1]=6.30843;
  par3[0][1]=9.08976e-06;

  par0[0][2]=0.00499917;
  par1[0][2]=0.290679;
  par2[0][2]=6.3041;
  par3[0][2]=7.38575e-05;

  par0[0][3]=0.00651151;
  par1[0][3]=0.256337;
  par2[0][3]=6.28936;
  par3[0][3]=2.57065e-05;

  par0[0][4]=0.00926919;
  par1[0][4]=0.251669;
  par2[0][4]=3.88212;
  par3[0][4]=3.90921e-06;

  par0[1][0]=0.000711682;
  par1[1][0]=0.714668;
  par2[1][0]=0.280541;
  par3[1][0]=1.28554e-05;

  par0[1][1]=0.00558565;
  par1[1][1]=0.358809;
  par2[1][1]=5.98924;
  par3[1][1]=2.01128e-05;

  par0[1][2]=0.00574133;
  par1[1][2]=0.30042;
  par2[1][2]=6.974;
  par3[1][2]=3.84421e-05;

  par0[1][3]=0.00582037;
  par1[1][3]=0.319711;
  par2[1][3]=5.07098;
  par3[1][3]=1.93082e-05;

  par0[1][4]=0.00607955;
  par1[1][4]=0.261425;
  par2[1][4]=6.80994;
  par3[1][4]=5.75288e-05;

  par0[2][0]=0.00250881;
  par1[2][0]=0.695641;
  par2[2][0]=2.65615;
  par3[2][0]=0.505847;

  par0[2][1]=0.0057382;
  par1[2][1]=0.532871;
  par2[2][1]=1.71524;
  par3[2][1]=3.11275;

  par0[2][2]=0.00558633;
  par1[2][2]=0.584366;
  par2[2][2]=3.92213;
  par3[2][2]=0.366339;

  par0[2][3]=0.00810919;
  par1[2][3]=0.571256;
  par2[2][3]=3.92309;
  par3[2][3]=0.434857;

  par0[2][4]=0.0118191;
  par1[2][4]=0.735462;
  par2[2][4]=-2.49231;
  par3[2][4]=4.85371;

  par0[3][0]=-0.0289151;
  par1[3][0]=4.85819;
  par2[3][0]=-8.47945;
  par3[3][0]=1.44712;

  par0[3][1]=-0.0180188;
  par1[3][1]=3.11055;
  par2[3][1]=-1.12097;
  par3[3][1]=0.0383473;

  par0[3][2]=-0.00637566;
  par1[3][2]=2.21049;
  par2[3][2]=0.673288;
  par3[3][2]=0.0256201;

  par0[3][3]=-0.00667959;
  par1[3][3]=2.26549;
  par2[3][3]=0.259882;
  par3[3][3]=0.112278;

  par0[3][4]=-0.0602224;
  par1[3][4]=9.99994;
  par2[3][4]=-32.8059;
  par3[3][4]=0.703987;

  par0[4][0]=0.0158817;
  par1[4][0]=0.496523;
  par2[4][0]=-0.0763621;
  par3[4][0]=4.97748;

  par0[4][1]=0.0139842;
  par1[4][1]=1.05686;
  par2[4][1]=-6.96373;
  par3[4][1]=4.99231;

  par0[4][2]=0.0209431;
  par1[4][2]=0.921923;
  par2[4][2]=-4.83827;
  par3[4][2]=4.99915;

  par0[4][3]=0.0212339;
  par1[4][3]=1.36597;
  par2[4][3]=-7.97446;
  par3[4][3]=4.04842;

  par0[4][4]=0.0685908;
  par1[4][4]=4.3994e-05;
  par2[4][4]=4.9234;
  par3[4][4]=1.94828;


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
  if (et>200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(200-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((200-par2[iEtaSl][iBremSl])*(200-par2[iEtaSl][iBremSl]));

  if (et>5 && et<200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}
double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_badtrack(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=5;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.4, 0.8, 1.44, 1.56, 2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8, 1.2, 1.4,  1.6,  2.5, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.0127153;
  par1[0][0]=0.208273;
  par2[0][0]=9.89164;
  par3[0][0]=8.55063e-05;

  par0[0][1]=0.0130164;
  par1[0][1]=0.162378;
  par2[0][1]=9.99907;
  par3[0][1]=0.263862;

  par0[0][2]=0.00976894;
  par1[0][2]=0.0907183;
  par2[0][2]=9.98796;
  par3[0][2]=0.125061;

  par0[0][3]=0.00801565;
  par1[0][3]=0.157897;
  par2[0][3]=9.45792;
  par3[0][3]=0.213528;

  par0[0][4]=0.0065934;
  par1[0][4]=0.47029;
  par2[0][4]=2.25598;
  par3[0][4]=6.34143e-06;

  par0[1][0]=-0.00011471;
  par1[1][0]=1.16998;
  par2[1][0]=-11.3343;
  par3[1][0]=0.0195891;

  par0[1][1]=-0.0138974;
  par1[1][1]=2.79251;
  par2[1][1]=-26.1356;
  par3[1][1]=3.45777;

  par0[1][2]=0.00294071;
  par1[1][2]=0.587785;
  par2[1][2]=2.54341;
  par3[1][2]=5;

  par0[1][3]=0.00947724;
  par1[1][3]=0.189705;
  par2[1][3]=9.78771;
  par3[1][3]=0.0738358;

  par0[1][4]=0.014365;
  par1[1][4]=0.354646;
  par2[1][4]=4.94541;
  par3[1][4]=0.181993;

  par0[2][0]=0.0107902;
  par1[2][0]=0.226565;
  par2[2][0]=8.2101;
  par3[2][0]=0.00472396;

  par0[2][1]=-0.00295697;
  par1[2][1]=1.72662;
  par2[2][1]=-23.9922;
  par3[2][1]=4.55417e-09;

  par0[2][2]=0.00112878;
  par1[2][2]=1.08996;
  par2[2][2]=-4.45665;
  par3[2][2]=0.667515;

  par0[2][3]=0.00910712;
  par1[2][3]=0.644996;
  par2[2][3]=2.69639;
  par3[2][3]=0.443691;

  par0[2][4]=0.0140811;
  par1[2][4]=1.66475;
  par2[2][4]=-4.36633;
  par3[2][4]=0.685134;

  par0[3][0]=0.00149322;
  par1[3][0]=0.0898527;
  par2[3][0]=-27.6767;
  par3[3][0]=8.99358e-05;

  par0[3][1]=-0.0635921;
  par1[3][1]=4.78013;
  par2[3][1]=10;
  par3[3][1]=5;

  par0[3][2]=-0.0506222;
  par1[3][2]=5.40357;
  par2[3][2]=-29.4451;
  par3[3][2]=0.0106964;

  par0[3][3]=-0.0589598;
  par1[3][3]=9.99904;
  par2[3][3]=-35.8349;
  par3[3][3]=4.0027;

  par0[3][4]=0.00582415;
  par1[3][4]=2.33571;
  par2[3][4]=-4.37696;
  par3[3][4]=0.219069;

  par0[4][0]=0.0104855;
  par1[4][0]=0.691632;
  par2[4][0]=2.63724;
  par3[4][0]=0.533841;

  par0[4][1]=0.0144748;
  par1[4][1]=0.637285;
  par2[4][1]=0.0120584;
  par3[4][1]=4.99764;

  par0[4][2]=0.0134224;
  par1[4][2]=0.921312;
  par2[4][2]=-2.81755;
  par3[4][2]=4.99812;

  par0[4][3]=0.0158206;
  par1[4][3]=1.24612;
  par2[4][3]=-5.49215;
  par3[4][3]=4.96863;

  par0[4][4]=0.013061;
  par1[4][4]=2.28987;
  par2[4][4]=-9.09829;
  par3[4][4]=0.00245111;


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
  if (et>200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(200-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((200-par2[iEtaSl][iBremSl])*(200-par2[iEtaSl][iBremSl]));

  if (et>5 && et<200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_showering(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=5;
  const Double_t EtaBins[nBinsEta+1]  = {0.0, 0.4, 0.8, 1.44, 1.56, 2.5};

  const int nBinsBrem=5;
  const Double_t BremBins[nBinsBrem+1]= {0.8, 1.2, 1.4,  1.6,  2.5, 8.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00920502;
  par1[0][0]=0.173937;
  par2[0][0]=6.34743;
  par3[0][0]=4.72947e-06;

  par0[0][1]=0.00415462;
  par1[0][1]=0.652331;
  par2[0][1]=-10.5185;
  par3[0][1]=8.60578e-08;

  par0[0][2]=0.00322326;
  par1[0][2]=0.677198;
  par2[0][2]=-9.96524;
  par3[0][2]=1.15228e-07;

  par0[0][3]=0.0080871;
  par1[0][3]=0.306903;
  par2[0][3]=2.79353;
  par3[0][3]=2.52121e-06;

  par0[0][4]=0.00641827;
  par1[0][4]=0.569942;
  par2[0][4]=0.597995;
  par3[0][4]=1.236e-06;

  par0[1][0]=0.00782689;
  par1[1][0]=0.263434;
  par2[1][0]=5.07523;
  par3[1][0]=6.89495e-06;

  par0[1][1]=0.00688063;
  par1[1][1]=0.416734;
  par2[1][1]=0.774306;
  par3[1][1]=1.88585e-06;

  par0[1][2]=0.007743;
  par1[1][2]=0.357;
  par2[1][2]=2.09378;
  par3[1][2]=2.17658e-06;

  par0[1][3]=0.00675108;
  par1[1][3]=0.419889;
  par2[1][3]=1.38645;
  par3[1][3]=2.46496e-06;

  par0[1][4]=0.00658488;
  par1[1][4]=0.549552;
  par2[1][4]=1.68595;
  par3[1][4]=2.18184e-06;

  par0[2][0]=0.00685344;
  par1[2][0]=0.413889;
  par2[2][0]=2.6537;
  par3[2][0]=4.00873;

  par0[2][1]=0.00557144;
  par1[2][1]=0.72999;
  par2[2][1]=-0.0955013;
  par3[2][1]=0.151207;

  par0[2][2]=0.00468369;
  par1[2][2]=0.857766;
  par2[2][2]=-2.40757;
  par3[2][2]=1.36861;

  par0[2][3]=0.00786699;
  par1[2][3]=0.816624;
  par2[2][3]=-1.64276;
  par3[2][3]=0.951636;

  par0[2][4]=0.00582809;
  par1[2][4]=1.75353;
  par2[2][4]=-8.13699;
  par3[2][4]=4.99062;

  par0[3][0]=0.039802;
  par1[3][0]=2.84046;
  par2[3][0]=-39.9938;
  par3[3][0]=4.71094;

  par0[3][1]=-0.0535086;
  par1[3][1]=9.99935;
  par2[3][1]=-28.795;
  par3[3][1]=0.487176;

  par0[3][2]=-0.0491194;
  par1[3][2]=9.9894;
  par2[3][2]=-30.145;
  par3[3][2]=4.99955;

  par0[3][3]=-0.0515311;
  par1[3][3]=9.99893;
  par2[3][3]=-34.9336;
  par3[3][3]=0.216566;

  par0[3][4]=-0.00808414;
  par1[3][4]=3.98766;
  par2[3][4]=-15.4094;
  par3[3][4]=0.732441;

  par0[4][0]=0.0145853;
  par1[4][0]=0.598377;
  par2[4][0]=9.62866;
  par3[4][0]=0.0793913;

  par0[4][1]=0.0195596;
  par1[4][1]=0.367598;
  par2[4][1]=3.0268;
  par3[4][1]=4.99248;

  par0[4][2]=0.0193685;
  par1[4][2]=0.615597;
  par2[4][2]=-0.665648;
  par3[4][2]=4.99816;

  par0[4][3]=0.0209289;
  par1[4][3]=1.00306;
  par2[4][3]=-4.07831;
  par3[4][3]=4.97443;

  par0[4][4]=0.0156225;
  par1[4][4]=2.00773;
  par2[4][4]=-7.48634;
  par3[4][4]=0.00173303;


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
  if (et>200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(200-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((200-par2[iEtaSl][iBremSl])*(200-par2[iEtaSl][iBremSl]));

  if (et>5 && et<200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}

double EnergyUncertaintyElectronSpecific::computeElectronEnergyUncertainty_crack(double eta, double brem, double energy){

  double et = energy/cosh(eta);

  const int nBinsEta=6;
  const double EtaBins[nBinsEta+1] = {0.0, 0.7, 1.15, 1.44, 1.56, 2.0, 2.5};

  const int nBinsBrem=6;
  const double BremBins  [nBinsBrem+1]= {0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0};

  float par0[nBinsEta][nBinsBrem];
  float par1[nBinsEta][nBinsBrem];
  float par2[nBinsEta][nBinsBrem];
  float par3[nBinsEta][nBinsBrem];

  par0[0][0]=0.00806753;
  par1[0][0]=0.143754;
  par2[0][0]=-0.00368104;
  par3[0][0]=0.219829;

  par0[0][1]=0.00899298;
  par1[0][1]=0.10159;
  par2[0][1]=4.70884;
  par3[0][1]=9.07419e-08;

  par0[0][2]=0;
  par1[0][2]=0;
  par2[0][2]=0;
  par3[0][2]=0;

  par0[0][3]=0;
  par1[0][3]=0;
  par2[0][3]=0;
  par3[0][3]=0;

  par0[0][4]=0;
  par1[0][4]=0;
  par2[0][4]=0;
  par3[0][4]=0;

  par0[0][5]=0;
  par1[0][5]=0;
  par2[0][5]=0;
  par3[0][5]=0;

  par0[1][0]=0.00880649;
  par1[1][0]=0.0716169;
  par2[1][0]=5.23856;
  par3[1][0]=0.00632907;

  par0[1][1]=0.00972275;
  par1[1][1]=0.0752675;
  par2[1][1]=3.35623;
  par3[1][1]=2.49397e-07;

  par0[1][2]=0;
  par1[1][2]=0;
  par2[1][2]=0;
  par3[1][2]=0;

  par0[1][3]=0;
  par1[1][3]=0;
  par2[1][3]=0;
  par3[1][3]=0;

  par0[1][4]=0;
  par1[1][4]=0;
  par2[1][4]=0;
  par3[1][4]=0;

  par0[1][5]=0;
  par1[1][5]=0;
  par2[1][5]=0;
  par3[1][5]=0;

  par0[2][0]=0.0101474;
  par1[2][0]=-0.332171;
  par2[2][0]=-31.8456;
  par3[2][0]=22.543;

  par0[2][1]=0.0109109;
  par1[2][1]=0.0425903;
  par2[2][1]=6.52561;
  par3[2][1]=2.18593e-08;

  par0[2][2]=0;
  par1[2][2]=0;
  par2[2][2]=0;
  par3[2][2]=0;

  par0[2][3]=0;
  par1[2][3]=0;
  par2[2][3]=0;
  par3[2][3]=0;

  par0[2][4]=0;
  par1[2][4]=0;
  par2[2][4]=0;
  par3[2][4]=0;

  par0[2][5]=0;
  par1[2][5]=0;
  par2[2][5]=0;
  par3[2][5]=0;

  par0[3][0]=0.00343003;
  par1[3][0]=11.5791;
  par2[3][0]=-112.084;
  par3[3][0]=-863.968;

  par0[3][1]=0.0372159;
  par1[3][1]=1.44028;
  par2[3][1]=-40;
  par3[3][1]=0.00102639;

  par0[3][2]=0;
  par1[3][2]=0;
  par2[3][2]=0;
  par3[3][2]=0;

  par0[3][3]=0;
  par1[3][3]=0;
  par2[3][3]=0;
  par3[3][3]=0;

  par0[3][4]=0;
  par1[3][4]=0;
  par2[3][4]=0;
  par3[3][4]=0;

  par0[3][5]=0;
  par1[3][5]=0;
  par2[3][5]=0;
  par3[3][5]=0;

  par0[4][0]=0.0192411;
  par1[4][0]=0.0511006;
  par2[4][0]=7.56304;
  par3[4][0]=0.00331583;

  par0[4][1]=0.0195124;
  par1[4][1]=0.104321;
  par2[4][1]=5.71476;
  par3[4][1]=6.12472e-06;

  par0[4][2]=0;
  par1[4][2]=0;
  par2[4][2]=0;
  par3[4][2]=0;

  par0[4][3]=0;
  par1[4][3]=0;
  par2[4][3]=0;
  par3[4][3]=0;

  par0[4][4]=0;
  par1[4][4]=0;
  par2[4][4]=0;
  par3[4][4]=0;

  par0[4][5]=0;
  par1[4][5]=0;
  par2[4][5]=0;
  par3[4][5]=0;

  par0[5][0]=0.0203644;
  par1[5][0]=-0.050789;
  par2[5][0]=-7.96854;
  par3[5][0]=4.71223;

  par0[5][1]=0.0198718;
  par1[5][1]=0.106859;
  par2[5][1]=3.54235;
  par3[5][1]=6.89631e-06;

  par0[5][2]=0;
  par1[5][2]=0;
  par2[5][2]=0;
  par3[5][2]=0;

  par0[5][3]=0;
  par1[5][3]=0;
  par2[5][3]=0;
  par3[5][3]=0;

  par0[5][4]=0;
  par1[5][4]=0;
  par2[5][4]=0;
  par3[5][4]=0;

  par0[5][5]=0;
  par1[5][5]=0;
  par2[5][5]=0;
  par3[5][5]=0;


  int iEtaSl = -1;
  for (int iEta = 0; iEta < nBinsEta; ++iEta){
    if ( EtaBins[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) <EtaBins[iEta+1] ){
      iEtaSl = iEta;
    }
  }

  int iBremSl = -1;
  for (int iBrem = 0; iBrem < nBinsBrem; ++iBrem){
      if ( BremBins[iBrem] <= brem && brem <BremBins[iBrem+1] ){
      iBremSl = iBrem;
    }
  }

  if (TMath::Abs(eta)>2.5) iEtaSl = nBinsEta-1;
  if (brem<BremBins[0]) iBremSl = 0;
  if (brem>BremBins[nBinsBrem-1]) iBremSl = nBinsBrem-1;
  if (brem>2) iBremSl = 1;

  float uncertainty = 0;
  if (et<5) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(5-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((5-par2[iEtaSl][iBremSl])*(5-par2[iEtaSl][iBremSl]));
  if (et>200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(200-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((200-par2[iEtaSl][iBremSl])*(200-par2[iEtaSl][iBremSl]));

  if (et>5 && et<200) uncertainty = par0[iEtaSl][iBremSl] + par1[iEtaSl][iBremSl]/(et-par2[iEtaSl][iBremSl]) + par3[iEtaSl][iBremSl]/((et-par2[iEtaSl][iBremSl])*(et-par2[iEtaSl][iBremSl]));

  return (uncertainty*energy);

}
