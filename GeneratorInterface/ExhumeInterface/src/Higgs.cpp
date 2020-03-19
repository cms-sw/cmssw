//-*-C++-*-
//-*-Higgs.cpp-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#include "GeneratorInterface/ExhumeInterface/interface/Higgs.h"
#include "GeneratorInterface/ExhumeInterface/interface/hdecay.h"

//#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"

extern "C" {
extern struct {
  int mdcy[3][500], mdme[2][8000];
  double brat[8000];
  int kfdp[5][8000];
} pydat3_;
}
#define pydat3 pydat3_

/////////////////////////////////////////////////////////////////////////////
Exhume::Higgs::Higgs(const edm::ParameterSet& pset) : CrossSection(pset) {
  std::cout << std::endl << "   = Higgs Subprocess Selected =" << std::endl;

  One = 1.0;
  BR = &One;
  HiggsMass2 = HiggsMass * HiggsMass;
  HiggsWidth = HiggsWidth_();
  TotWidth = HiggsWidth;

  SetC();

  /*
    double Hm2pGam2 = HiggsMass2 + HiggsWidth * HiggsWidth;

    double SqrtHm2pGam2 = sqrt(Hm2pGam2);
    double HmpSqrtHm2pGam2 = HiggsMass + SqrtHm2pGam2;
    //C normalises so that integral of |prop|^2 d(m^2) = 1

    C = (2.0/HiggsMass) * M_PI *SqrtHm2pGam2 *
      (HiggsWidth*HiggsWidth + HiggsMass * HmpSqrtHm2pGam2) / 
      (2.0 * pow(HiggsMass, 4.5) * HiggsWidth * sqrt(2 * HmpSqrtHm2pGam2));

    C = 1.0/sqrt(C);
    */
  //GGHConst = sqrt((*BR)) * 0.25 * I * LambdaW / M_PI;
  FsfTop = 0.25 / (TopMass * TopMass);
  FsfBottom = 0.25 / (BottomMass * BottomMass);

  //Gev2fb = 3.88 * pow(10.0,11);
  NLOConst = M_PI + 5.5 / M_PI;

  Name = "Higgs";
  Partons.resize(1);
  Partons[0].id = 25;
}
//////////////////////////////////////////////////////////////////////////////
inline void Exhume::Higgs::SetC() {
  double Hm2pGam2 = HiggsMass2 + HiggsWidth * HiggsWidth;
  double SqrtHm2pGam2 = sqrt(Hm2pGam2);
  double HmpSqrtHm2pGam2 = HiggsMass + SqrtHm2pGam2;

  C = (2.0 / HiggsMass) * M_PI * SqrtHm2pGam2 * (HiggsWidth * HiggsWidth + HiggsMass * HmpSqrtHm2pGam2) /
      (2.0 * pow(HiggsMass, 4.5) * HiggsWidth * sqrt(2 * HmpSqrtHm2pGam2));

  C = sqrt(1.0 / C);

  GGHConst = sqrt((*BR)) * 0.25 * I * LambdaW / M_PI;
}
//////////////////////////////////////////////////////////////////////////////
void Exhume::Higgs::SetHiggsMass(const double& HM_) {
  HiggsMass = HM_;
  HiggsMass2 = HiggsMass * HiggsMass;
  HiggsWidth = HiggsWidth_();
  TotWidth = HiggsWidth;
  HiggsWidth = TotWidth * (*BR);

  SetC();

  /*
  double Hm2pGam2 = HiggsMass2 + HiggsWidth * HiggsWidth;
  
  C = 0.25*M_PI*pow(Hm2pGam2,0.25)*
    (2.0*HiggsWidth*HiggsWidth + 3.0*HiggsMass2 - 
     HiggsMass*sqrt(Hm2pGam2))/(HiggsWidth*pow(HiggsMass,4.5));
  
  C = 1.0/sqrt(C);
  */
  return;
}
/////////////////////////////////////////////////////////////////////////////
double Exhume::Higgs::SubProcess() {
  AlphaS_ = AlphaS(SqrtsHat);

  double ModAmp = abs(GluGlu2HiggsAmp() * Propagator() * C);

  return ((1.0 + AlphaS_ * NLOConst) * (Gev2fb * ModAmp * ModAmp * M_PI * InvsHat2));
}
/////////////////////////////////////////////////////////////////////////////
double Exhume::Higgs::HiggsWidth_() {
  //  std::cout<<std::endl<<
  //"  Using HDECAY to calculate S.M. Higgs Width"<<std::endl;

  flag_.ihiggs = 0;
  flag_.nnlo = 0;
  flag_.ipole = 1;
  oldfash_.nfgg = 5;
  onshell_.ionsh = 0;
  onshell_.ionwz = 0;
  param_.gf = 1.16639e-5;
  param_.alph = AlphaEw;  //1.0/137.0;
  param_.amtau = 1.7771;  //tau_mass;//1.7771;
  param_.ammuon = MuonMass;
  param_.amz = ZMass;  //91.187;
  param_.amw = WMass;  //80.33;
  ckmpar_.vus = 0.2205;
  ckmpar_.vcb = 0.04;
  ckmpar_.vub = 0.08 * ckmpar_.vcb;
  masses_.ams = StrangeMass;
  masses_.amc = 1.42;        //charm_mass;//1.42;
  masses_.amb = BottomMass;  //4.62;
  masses_.amt = TopMass;     //175.0;
  strange_.amsb = masses_.ams;
  als_.amc0 = masses_.amc;
  als_.amb0 = masses_.amb;
  als_.amt0 = masses_.amt;
  double acc = 1.0e-8;
  int nloop = 2;
  double alsmz = 0.118;

  als_.xlambda = xitla_(&nloop, &alsmz, &acc);
  als_.n0 = 5;
  alsini_(&acc);
  wzwdth_.gamw = 2.08;
  wzwdth_.gamz = 2.49;

  int nber = 18;
  bernini_(&nber);
  hmass_.amsm = HiggsMass;
  double GfAmt = param_.gf * masses_.amt;
  double Mw_Mt2 = (param_.amw * param_.amw) / (masses_.amt * masses_.amt);
  double one_Mw_Mt2 = 1.0 - Mw_Mt2;
  wzwdth_.gamt0 = GfAmt * GfAmt * GfAmt * 0.125 / sqrt(2.0) / M_PI * one_Mw_Mt2 * one_Mw_Mt2 * (1 + 2 * Mw_Mt2);
  wzwdth_.gamt1 = wzwdth_.gamt0;
  //double tgbet = 1.0;
  //hdec_(&tgbet);
  hdec_();

  //std::cout<<std::endl<<"  S.M. Higgs width = "<<widthsm_.smwdth<<std::endl;

  return (widthsm_.smwdth);
}
/////////////////////////////////////////////////////////////////////////////
void Exhume::Higgs::SetPartons() {
  Partons[0].p = CentralVector;

  return;
}
/////////////////////////////////////////////////////////////////////////////
inline double Exhume::Higgs::SubParameterWeight() { return (1.0); }
/////////////////////////////////////////////////////////////////////////////
inline double Exhume::Higgs::SubParameterRange() { return (1.0); }
/////////////////////////////////////////////////////////////////////////////
void Exhume::Higgs::MaximiseSubParameters() { return; }
/////////////////////////////////////////////////////////////////////////////
void Exhume::Higgs::SetSubParameters() { return; }
/////////////////////////////////////////////////////////////////////////////
void Exhume::Higgs::SetHiggsDecay(const int& id_) {
  pydat3.mdme[0][209] = 0;
  pydat3.mdme[0][210] = 0;
  pydat3.mdme[0][211] = 0;
  pydat3.mdme[0][212] = 0;
  pydat3.mdme[0][213] = 0;
  pydat3.mdme[0][214] = 0;
  pydat3.mdme[0][215] = 0;
  pydat3.mdme[0][216] = 0;
  pydat3.mdme[0][217] = 0;
  pydat3.mdme[0][218] = 0;
  pydat3.mdme[0][219] = 0;
  pydat3.mdme[0][220] = 0;
  pydat3.mdme[0][221] = 0;
  pydat3.mdme[0][222] = 0;
  pydat3.mdme[0][223] = 0;
  pydat3.mdme[0][224] = 0;
  pydat3.mdme[0][225] = 0;

  //double BR;

  switch (id_) {
    case 3:
      //strange
      pydat3.mdme[0][211] = 1;
      BR = &widthsm_.smbrs;
      break;
    case 4:
      //charm
      pydat3.mdme[0][212] = 1;
      BR = &widthsm_.smbrc;
      break;
    case 5:
      //bottom
      pydat3.mdme[0][213] = 1;
      BR = &widthsm_.smbrb;
      break;
    case 6:
      //top
      pydat3.mdme[0][214] = 1;
      BR = &widthsm_.smbrt;
      break;
    case 13:
      //muon
      pydat3.mdme[0][218] = 1;
      BR = &widthsm_.smbrm;
      break;
    case 15:
      //tau
      pydat3.mdme[0][219] = 1;
      BR = &widthsm_.smbrl;
      break;
    case 21:
      //gluon
      pydat3.mdme[0][221] = 1;
      BR = &widthsm_.smbrg;
      break;
    case 22:
      //photon
      pydat3.mdme[0][222] = 1;
      BR = &widthsm_.smbrga;
      break;
    case 23:
      //Z
      pydat3.mdme[0][224] = 1;
      BR = &widthsm_.smbrz;
      break;
    case 24:
      //W
      pydat3.mdme[0][225] = 1;
      BR = &widthsm_.smbrw;
      break;

    default:
      std::cout << "   Unrecognised PDG code for Higgs Decay" << std::endl;
      for (int alldecay = 209; alldecay < 226; alldecay++) {
        pydat3.mdme[0][alldecay] = 1;
      }
      BR = &One;  //1.0;
      break;
  }

  //HiggsWidth = TotWidth * (*BR);

  SetC();

  /*

  double Hm2pGam2 = HiggsMass2 + HiggsWidth * HiggsWidth;
  double SqrtHm2pGam2 = sqrt(Hm2pGam2);
  double HmpSqrtHm2pGam2 = HiggsMass + SqrtHm2pGam2;

  C = (2.0/HiggsMass) * M_PI *SqrtHm2pGam2 *
    (HiggsWidth*HiggsWidth + HiggsMass * HmpSqrtHm2pGam2) / 
    (2.0 * pow(HiggsMass, 4.5) * HiggsWidth * sqrt(2 * HmpSqrtHm2pGam2));

  C = sqrt((*BR)/C);
  */
  return;
}
/////////////////////////////////////////////////////////////////////////////
