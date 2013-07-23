// $Id: MuonResiduals6DOFFitter.cc,v 1.9 2011/10/12 23:44:11 khotilov Exp $

#ifdef STANDALONE_FITTER
#include "MuonResiduals6DOFFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFFitter.h"
#endif

#include "TH2F.h"
#include "TMath.h"
#include "TTree.h"
#include "TFile.h"

namespace
{
  TMinuit *minuit;

  double sum_of_weights;
  double number_of_hits;
  bool weight_alignment;

  double residual_x(double delta_x, double delta_y, double delta_z,
                    double delta_phix, double delta_phiy, double delta_phiz,
                    double track_x, double track_y, double track_dxdz, double track_dydz,
                    double alphax, double residual_dxdz)
  {
    return delta_x
        - track_dxdz * delta_z
        - track_y * track_dxdz * delta_phix
        + track_x * track_dxdz * delta_phiy
        - track_y * delta_phiz
        + residual_dxdz * alphax;
  }

  double residual_y(double delta_x, double delta_y, double delta_z,
                    double delta_phix, double delta_phiy, double delta_phiz,
                    double track_x, double track_y, double track_dxdz, double track_dydz,
                    double alphay, double residual_dydz)
  {
    return delta_y
        - track_dydz * delta_z
        - track_y * track_dydz * delta_phix
        + track_x * track_dydz * delta_phiy
        + track_x * delta_phiz
        + residual_dydz * alphay;
  }

  double residual_dxdz(double delta_x, double delta_y, double delta_z,
                       double delta_phix, double delta_phiy, double delta_phiz,
                       double track_x, double track_y, double track_dxdz, double track_dydz)
  {
    return -track_dxdz * track_dydz * delta_phix
        + (1. + track_dxdz * track_dxdz) * delta_phiy
        - track_dydz * delta_phiz;
  }

  double residual_dydz(double delta_x, double delta_y, double delta_z,
                       double delta_phix, double delta_phiy, double delta_phiz,
                       double track_x, double track_y, double track_dxdz, double track_dydz)
  {
    return -(1. + track_dydz * track_dydz) * delta_phix
        + track_dxdz * track_dydz * delta_phiy
        + track_dxdz * delta_phiz;
  }

  Double_t residual_x_trackx_TF1(Double_t *xx, Double_t *p)    { return 10.*residual_x(p[0], p[1], p[2], p[3], p[4], p[5], xx[0], p[7],  p[8],  p[9],  p[10], p[11]); }
  Double_t residual_x_tracky_TF1(Double_t *xx, Double_t *p)    { return 10.*residual_x(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  xx[0], p[8],  p[9],  p[10], p[11]); }
  Double_t residual_x_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  xx[0], p[9],  p[10], p[11]); }
  Double_t residual_x_trackdydz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  p[8],  xx[0], p[10], p[11]); }

  Double_t residual_y_trackx_TF1(Double_t *xx, Double_t *p)    { return 10.*residual_y(p[0], p[1], p[2], p[3], p[4], p[5], xx[0], p[7], p[8], p[9], p[12], p[13]); }
  Double_t residual_y_tracky_TF1(Double_t *xx, Double_t *p)    { return 10.*residual_y(p[0], p[1], p[2], p[3], p[4], p[5], p[6], xx[0], p[8], p[9], p[12], p[13]); }
  Double_t residual_y_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_y(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], xx[0], p[9], p[12], p[13]); }
  Double_t residual_y_trackdydz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_y(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], xx[0], p[12], p[13]); }

  Double_t residual_dxdz_trackx_TF1(Double_t *xx, Double_t *p)    { return 1000.*residual_dxdz(p[0], p[1], p[2], p[3], p[4], p[5], xx[0], p[7],  p[8],  p[9]); }
  Double_t residual_dxdz_tracky_TF1(Double_t *xx, Double_t *p)    { return 1000.*residual_dxdz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  xx[0], p[8],  p[9]); }
  Double_t residual_dxdz_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  xx[0], p[9]); }
  Double_t residual_dxdz_trackdydz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  p[8],  xx[0]); }

  Double_t residual_dydz_trackx_TF1(Double_t *xx, Double_t *p)    { return 1000.*residual_dydz(p[0], p[1], p[2], p[3], p[4], p[5], xx[0], p[7],  p[8],  p[9]); }
  Double_t residual_dydz_tracky_TF1(Double_t *xx, Double_t *p)    { return 1000.*residual_dydz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  xx[0], p[8],  p[9]); }
  Double_t residual_dydz_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dydz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  xx[0], p[9]); }
  Double_t residual_dydz_trackdydz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dydz(p[0], p[1], p[2], p[3], p[4], p[5], p[6],  p[7],  p[8],  xx[0]); }
}


void MuonResiduals6DOFFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag)
{
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(minuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter)
  {
    const double residX = (*resiter)[MuonResiduals6DOFFitter::kResidX];
    const double residY = (*resiter)[MuonResiduals6DOFFitter::kResidY];
    const double resslopeX = (*resiter)[MuonResiduals6DOFFitter::kResSlopeX];
    const double resslopeY = (*resiter)[MuonResiduals6DOFFitter::kResSlopeY];
    const double positionX = (*resiter)[MuonResiduals6DOFFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFFitter::kRedChi2];

    const double alignx = par[MuonResiduals6DOFFitter::kAlignX];
    const double aligny = par[MuonResiduals6DOFFitter::kAlignY];
    const double alignz = par[MuonResiduals6DOFFitter::kAlignZ];
    const double alignphix = par[MuonResiduals6DOFFitter::kAlignPhiX];
    const double alignphiy = par[MuonResiduals6DOFFitter::kAlignPhiY];
    const double alignphiz = par[MuonResiduals6DOFFitter::kAlignPhiZ];
    const double resXsigma = par[MuonResiduals6DOFFitter::kResidXSigma];
    const double resYsigma = par[MuonResiduals6DOFFitter::kResidYSigma];
    const double slopeXsigma = par[MuonResiduals6DOFFitter::kResSlopeXSigma];
    const double slopeYsigma = par[MuonResiduals6DOFFitter::kResSlopeYSigma];
    const double alphax = par[MuonResiduals6DOFFitter::kAlphaX];
    const double alphay = par[MuonResiduals6DOFFitter::kAlphaY];
    const double resXgamma = par[MuonResiduals6DOFFitter::kResidXGamma];
    const double resYgamma = par[MuonResiduals6DOFFitter::kResidYGamma];
    const double slopeXgamma = par[MuonResiduals6DOFFitter::kResSlopeXGamma];
    const double slopeYgamma = par[MuonResiduals6DOFFitter::kResSlopeYGamma];

    double coefX = alphax, coefY = alphay;
    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian ||
        fitter->residualsModel() == MuonResidualsFitter::kPureGaussian2D) coefX = coefY = 0.;
    double residXpeak = residual_x(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, coefX, resslopeX);
    double residYpeak = residual_y(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, coefY, resslopeY);
    double slopeXpeak = residual_dxdz(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY);
    double slopeYpeak = residual_dydz(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY);

    double weight = (1./redchi2) * number_of_hits / sum_of_weights;
    if (!weight_alignment) weight = 1.;

    if (!weight_alignment  ||  TMath::Prob(redchi2*12, 12) < 0.99)  // no spikes allowed
    {
      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
        if (fitter->useRes() == MuonResidualsFitter::k1111) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1110) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1100) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k0010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
        }
        //std::cout<<"FCNx("<<residX<<","<<residXpeak<<","<<resXsigma<<") = "<<MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma)<<std::endl;
        //std::cout<<"FCNy("<<residY<<","<<residYpeak<<","<<resYsigma<<") = "<<MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma)<<std::endl;
        //std::cout<<"FCNsx("<<resslopeX<<","<<slopeXpeak<<","<<slopeXsigma<<") = "<<MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)<<std::endl;
        //std::cout<<"FCNsy("<<resslopeY<<","<<slopeYpeak<<","<<slopeYsigma<<") = "<<MuonResidualsFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma)<<std::endl;
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian2D) {
        if (fitter->useRes() == MuonResidualsFitter::k1111) {
          fval += -weight * MuonResidualsFitter_logPureGaussian2D(residX, resslopeX, residXpeak, slopeXpeak, resXsigma, slopeXsigma, alphax);
          fval += -weight * MuonResidualsFitter_logPureGaussian2D(residY, resslopeY, residYpeak, slopeYpeak, resYsigma, slopeYsigma, alphay);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1110) {
          fval += -weight * MuonResidualsFitter_logPureGaussian2D(residX, resslopeX, residXpeak, slopeXpeak, resXsigma, slopeXsigma, alphax);
          fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1100) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian2D(residX, resslopeX, residXpeak, slopeXpeak, resXsigma, slopeXsigma, alphax);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k0010) {
                fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
        }
        //std::cout<<"FCNx("<<residX<<","<<resslopeX<<","<<residXpeak<<","<<slopeXpeak<<","<<resXsigma<<","<<slopeXsigma<<","<<alphax<<") = "<<MuonResidualsFitter_logPureGaussian2D(residX, resslopeX, residXpeak, slopeXpeak, resXsigma, slopeXsigma, alphax)<<std::endl;
        //std::cout<<"FCNy("<<residY<<","<<resslopeY<<","<<residYpeak<<","<<slopeYpeak<<","<<resYsigma<<","<<slopeYsigma<<","<<alphay<<") = "<<MuonResidualsFitter_logPureGaussian2D(residY, resslopeY, residYpeak, slopeYpeak, resYsigma, slopeYsigma, alphay)<<std::endl;
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
        fval += -weight * MuonResidualsFitter_logPowerLawTails(residX, residXpeak, resXsigma, resXgamma);
        fval += -weight * MuonResidualsFitter_logPowerLawTails(residY, residYpeak, resYsigma, resYgamma);
        fval += -weight * MuonResidualsFitter_logPowerLawTails(resslopeX, slopeXpeak, slopeXsigma, slopeXgamma);
        fval += -weight * MuonResidualsFitter_logPowerLawTails(resslopeY, slopeYpeak, slopeYsigma, slopeYgamma);
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
        fval += -weight * MuonResidualsFitter_logROOTVoigt(residX, residXpeak, resXsigma, resXgamma);
        fval += -weight * MuonResidualsFitter_logROOTVoigt(residY, residYpeak, resYsigma, resYgamma);
        fval += -weight * MuonResidualsFitter_logROOTVoigt(resslopeX, slopeXpeak, slopeXsigma, slopeXgamma);
        fval += -weight * MuonResidualsFitter_logROOTVoigt(resslopeY, slopeYpeak, slopeYsigma, slopeYgamma);
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(residX, residXpeak, resXsigma);
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(residY, residYpeak, resYsigma);
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(resslopeX, slopeXpeak, slopeXsigma);
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(resslopeY, slopeYpeak, slopeYsigma);
      }
      else { assert(false); }
    }
  }
/*
  const double resXsigma = par[MuonResiduals6DOFFitter::kResidXSigma];
  const double resYsigma = par[MuonResiduals6DOFFitter::kResidYSigma];
  const double slopeXsigma = par[MuonResiduals6DOFFitter::kResSlopeXSigma];
  const double slopeYsigma = par[MuonResiduals6DOFFitter::kResSlopeYSigma];
  const double alphax = par[MuonResiduals6DOFFitter::kAlphaX];
  const double alphay = par[MuonResiduals6DOFFitter::kAlphaY];
  std::cout<<"fval="<<fval<<","<<resXsigma<<","<<slopeXsigma<<","<<alphax<<","<<resYsigma<<","<<slopeYsigma<<","<<alphay<<std::endl;
*/
}


void MuonResiduals6DOFFitter::inform(TMinuit *tMinuit)
{
  minuit = tMinuit;
}


void MuonResiduals6DOFFitter::correctBField()
{
  MuonResidualsFitter::correctBField(kPt, kCharge);
}


double MuonResiduals6DOFFitter::sumofweights()
{
  sum_of_weights = 0.;
  number_of_hits = 0.;
  weight_alignment = m_weightAlignment;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    if (m_weightAlignment) {
      double redchi2 = (*resiter)[kRedChi2];
      if (TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed
        sum_of_weights += 1./redchi2;
        number_of_hits += 1.;
      }
    }
    else {
      sum_of_weights += 1.;
      number_of_hits += 1.;
    }
  }
  return sum_of_weights;
}


bool MuonResiduals6DOFFitter::fit(Alignable *ali)
{
  initialize_table();  // if not already initialized
  sumofweights();

  double resx_std = 0.5;
  double resy_std = 3.0;
  double resslopex_std = 0.002;
  double resslopey_std = 0.005;

  int nums[16]          = {kAlignX, kAlignY, kAlignZ, kAlignPhiX, kAlignPhiY, kAlignPhiZ,   kResidXSigma, kResidYSigma, kResSlopeXSigma, kResSlopeYSigma,
                           kAlphaX, kAlphaY,    kResidXGamma, kResidYGamma, kResSlopeXGamma, kResSlopeYGamma};
  std::string names[16] = {"AlignX","AlignY","AlignZ","AlignPhiX","AlignPhiY","AlignPhiZ",  "ResidXSigma","ResidYSigma","ResSlopeXSigma","ResSlopeYSigma",
                           "AlphaX","AlphaY",   "ResidXGamma","ResidYGamma","ResSlopeXGamma","ResSlopeYGamma"};
  double starts[16]     = {0., 0., 0., 0., 0., 0.,    resx_std, resy_std, resslopex_std, resslopey_std,
                           0., 0.,    0.1*resx_std, 0.1*resy_std, 0.1*resslopex_std, 0.1*resslopey_std};
  double steps[16]      = {0.1, 0.1, 0.1, 0.001, 0.001, 0.001,   0.001*resx_std, 0.001*resy_std, 0.001*resslopex_std, 0.001*resslopey_std,
                           0.001, 0.001,    0.01*resx_std, 0.01*resy_std, 0.01*resslopex_std, 0.01*resslopey_std};
  double lows[16]       = {0., 0., 0., 0., 0., 0.,    0., 0., 0., 0.,
                           -1., -1.,    0., 0., 0., 0.};
  double highs[16]      = {0., 0., 0., 0., 0., 0.,    10., 10., 0.1, 0.1,
                           1.,1.,    0., 0., 0., 0.};

  std::vector<int> num(nums, nums+6);
  std::vector<std::string> name(names, names+6);
  std::vector<double> start(starts, starts+6);
  std::vector<double> step(steps, steps+6);
  std::vector<double> low(lows, lows+6);
  std::vector<double> high(highs, highs+6);

  bool add_alpha = ( residualsModel() == kPureGaussian2D );
  bool add_gamma = ( residualsModel() == kROOTVoigt || residualsModel() == kPowerLawTails );

  int idx[8], ni = 0;
  if (useRes() == k1111) {
    for(ni=0; ni<4; ni++) idx[ni] = ni+6;
    if (add_alpha) for(; ni<6; ni++) idx[ni] = ni+6;
    else if (add_gamma) for(; ni<8; ni++) idx[ni] = ni+8;
    if (!add_alpha) fix(kAlphaX);
    if (!add_alpha) fix(kAlphaY);
  }
  else if (useRes() == k1110) {
    for(ni=0; ni<3; ni++) idx[ni] = ni+6;
    if (add_alpha) idx[ni++] = 10;
    else if (add_gamma) for(; ni<6; ni++) idx[ni] = ni+9;
    fix(kResSlopeYSigma);
    fix(kAlphaY);
    if (!add_alpha) fix(kAlphaX);
  }
  else if (useRes() == k1100) {
    for(ni=0; ni<2; ni++) idx[ni] = ni+6;
    if (add_gamma) for(; ni<4; ni++) idx[ni] = ni+10;
    fix(kResSlopeXSigma);
    fix(kResSlopeYSigma);
    fix(kAlphaX);
    fix(kAlphaY);
  }
  else if (useRes() == k1010) {
    idx[ni++] = 6; idx[ni++] = 8;
    if (add_alpha) idx[ni++] = 10;
    if (add_gamma) { idx[ni++] = 12; idx[ni++] = 14; }
    fix(kResidYSigma);
    fix(kResSlopeYSigma);
    fix(kAlphaY);
    if (!add_alpha) fix(kAlphaX);
  }
  else if (useRes() == k0010) {
    idx[ni++] = 8;
    if (add_gamma) idx[ni++] = 14;
    fix(kResidXSigma);
    fix(kResidYSigma);
    fix(kResSlopeYSigma);
    fix(kAlphaX);
    fix(kAlphaY);
  }
  for (int i=0; i<ni; i++){
    num.push_back(nums[idx[i]]);
    name.push_back(names[idx[i]]);
    start.push_back(starts[idx[i]]);
    step.push_back(steps[idx[i]]);
    low.push_back(lows[idx[i]]);
    high.push_back(highs[idx[i]]);
  }

  return dofit(&MuonResiduals6DOFFitter_FCN, num, name, start, step, low, high);
}


double MuonResiduals6DOFFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali)
{
  sumofweights();

  double mean_residualx = 0., mean_residualy = 0., mean_resslopex = 0., mean_resslopey = 0.;
  double mean_trackx = 0., mean_tracky = 0., mean_trackdxdz = 0., mean_trackdydz = 0.;
  double sum_w = 0.;

  for (std::vector<double*>::const_iterator rit = residuals_begin();  rit != residuals_end();  ++rit)
  {
    const double redchi2 = (*rit)[kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*12, 12) < 0.99)  // no spikes allowed
    {
      double factor_w = 1./(sum_w + weight);
      mean_residualx = factor_w * (sum_w * mean_residualx + weight * (*rit)[kResidX]);
      mean_residualy = factor_w * (sum_w * mean_residualy + weight * (*rit)[kResidY]);
      mean_resslopex = factor_w * (sum_w * mean_resslopex + weight * (*rit)[kResSlopeX]);
      mean_resslopey = factor_w * (sum_w * mean_resslopey + weight * (*rit)[kResSlopeY]);
      mean_trackx    = factor_w * (sum_w * mean_trackx    + weight * (*rit)[kPositionX]);
      mean_tracky    = factor_w * (sum_w * mean_tracky    + weight * (*rit)[kPositionY]);
      mean_trackdxdz = factor_w * (sum_w * mean_trackdxdz + weight * (*rit)[kAngleX]);
      mean_trackdydz = factor_w * (sum_w * mean_trackdydz + weight * (*rit)[kAngleY]);
      sum_w += weight;
    }
  }

  std::string name_x, name_y, name_dxdz, name_dydz, name_x_raw, name_y_raw, name_dxdz_raw, name_dydz_raw, name_x_cut, name_y_cut, name_alphax, name_alphay;
  std::string name_x_trackx, name_y_trackx, name_dxdz_trackx, name_dydz_trackx;
  std::string name_x_tracky, name_y_tracky, name_dxdz_tracky, name_dydz_tracky;
  std::string name_x_trackdxdz, name_y_trackdxdz, name_dxdz_trackdxdz, name_dydz_trackdxdz;
  std::string name_x_trackdydz, name_y_trackdydz, name_dxdz_trackdydz, name_dydz_trackdydz;

  name_x = name + "_x";
  name_y = name + "_y";
  name_dxdz = name + "_dxdz";
  name_dydz = name + "_dydz";
  name_x_raw = name + "_x_raw";
  name_y_raw = name + "_y_raw";
  name_dxdz_raw = name + "_dxdz_raw";
  name_dydz_raw = name + "_dydz_raw";
  name_x_cut = name + "_x_cut";
  name_y_cut = name + "_y_cut";
  name_alphax = name + "_alphax";
  name_alphay = name + "_alphay";
  name_x_trackx = name + "_x_trackx";
  name_y_trackx = name + "_y_trackx";
  name_dxdz_trackx = name + "_dxdz_trackx";
  name_dydz_trackx = name + "_dydz_trackx";
  name_x_tracky = name + "_x_tracky";
  name_y_tracky = name + "_y_tracky";
  name_dxdz_tracky = name + "_dxdz_tracky";
  name_dydz_tracky = name + "_dydz_tracky";
  name_x_trackdxdz = name + "_x_trackdxdz";
  name_y_trackdxdz = name + "_y_trackdxdz";
  name_dxdz_trackdxdz = name + "_dxdz_trackdxdz";
  name_dydz_trackdxdz = name + "_dydz_trackdxdz";
  name_x_trackdydz = name + "_x_trackdydz";
  name_y_trackdydz = name + "_y_trackdydz";
  name_dxdz_trackdydz = name + "_dxdz_trackdydz";
  name_dydz_trackdydz = name + "_dydz_trackdydz";

  double width = ali->surface().width();
  double length = ali->surface().length();
  int bins = 200;
  double min_x = -100.,            max_x = 100.;
  double min_y = -100.,            max_y = 100.;
  double min_dxdz = -75.,         max_dxdz = 75.;
  double min_dydz = -150.,         max_dydz = 150.;
  double min_trackx = -width/2.,   max_trackx = width/2.;
  double min_tracky = -length/2.,  max_tracky = length/2.;
  double min_trackdxdz = -1.5,     max_trackdxdz = 1.5;
  double min_trackdydz = -1.5,     max_trackdydz = 1.5;

  TH1F *hist_x = dir->make<TH1F>(name_x.c_str(), "", bins, min_x, max_x);
  TH1F *hist_y = dir->make<TH1F>(name_y.c_str(), "", bins, min_y, max_y);
  TH1F *hist_dxdz = dir->make<TH1F>(name_dxdz.c_str(), "", bins, min_dxdz, max_dxdz);
  TH1F *hist_dydz = dir->make<TH1F>(name_dydz.c_str(), "", bins, min_dydz, max_dydz);
  TH1F *hist_x_raw = dir->make<TH1F>(name_x_raw.c_str(), "", bins, min_x, max_x);
  TH1F *hist_y_raw = dir->make<TH1F>(name_y_raw.c_str(), "", bins, min_y, max_y);
  TH1F *hist_dxdz_raw = dir->make<TH1F>(name_dxdz_raw.c_str(), "", bins, min_dxdz, max_dxdz);
  TH1F *hist_dydz_raw = dir->make<TH1F>(name_dydz_raw.c_str(), "", bins, min_dydz, max_dydz);
  TH1F *hist_x_cut = dir->make<TH1F>(name_x_cut.c_str(), "", bins, min_x, max_x);
  TH1F *hist_y_cut = dir->make<TH1F>(name_y_cut.c_str(), "", bins, min_y, max_y);
  TH2F *hist_alphax = dir->make<TH2F>(name_alphax.c_str(), "", 50, 50, 50, 50, -50., 50.);
  TH2F *hist_alphay = dir->make<TH2F>(name_alphay.c_str(), "", 75, 100, 100, 75, -75., 75.);

  TProfile *hist_x_trackx = dir->make<TProfile>(name_x_trackx.c_str(), "", 50, min_trackx, max_trackx, min_x, max_x);
  TProfile *hist_y_trackx = dir->make<TProfile>(name_y_trackx.c_str(), "", 50, min_trackx, max_trackx, min_y, max_y);
  TProfile *hist_dxdz_trackx = dir->make<TProfile>(name_dxdz_trackx.c_str(), "", 50, min_trackx, max_trackx, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackx = dir->make<TProfile>(name_dydz_trackx.c_str(), "", 50, min_trackx, max_trackx, min_dydz, max_dydz);
  TProfile *hist_x_tracky = dir->make<TProfile>(name_x_tracky.c_str(), "", 50, min_tracky, max_tracky, min_x, max_x);
  TProfile *hist_y_tracky = dir->make<TProfile>(name_y_tracky.c_str(), "", 50, min_tracky, max_tracky, min_y, max_y);
  TProfile *hist_dxdz_tracky = dir->make<TProfile>(name_dxdz_tracky.c_str(), "", 50, min_tracky, max_tracky, min_dxdz, max_dxdz);
  TProfile *hist_dydz_tracky = dir->make<TProfile>(name_dydz_tracky.c_str(), "", 50, min_tracky, max_tracky, min_dydz, max_dydz);
  TProfile *hist_x_trackdxdz = dir->make<TProfile>(name_x_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_x, max_x);
  TProfile *hist_y_trackdxdz = dir->make<TProfile>(name_y_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_y, max_y);
  TProfile *hist_dxdz_trackdxdz = dir->make<TProfile>(name_dxdz_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackdxdz = dir->make<TProfile>(name_dydz_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_dydz, max_dydz);
  TProfile *hist_x_trackdydz = dir->make<TProfile>(name_x_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_x, max_x);
  TProfile *hist_y_trackdydz = dir->make<TProfile>(name_y_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_y, max_y);
  TProfile *hist_dxdz_trackdydz = dir->make<TProfile>(name_dxdz_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackdydz = dir->make<TProfile>(name_dydz_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_dydz, max_dydz);

  hist_x_trackx->SetAxisRange(-10., 10., "Y");
  hist_y_trackx->SetAxisRange(-20., 20., "Y");
  hist_dxdz_trackx->SetAxisRange(-10., 10., "Y");
  hist_dydz_trackx->SetAxisRange(-20., 20., "Y");
  hist_x_tracky->SetAxisRange(-10., 10., "Y");
  hist_y_tracky->SetAxisRange(-20., 20., "Y");
  hist_dxdz_tracky->SetAxisRange(-10., 10., "Y");
  hist_dydz_tracky->SetAxisRange(-20., 20., "Y");
  hist_x_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_y_trackdxdz->SetAxisRange(-20., 20., "Y");
  hist_dxdz_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_dydz_trackdxdz->SetAxisRange(-20., 20., "Y");
  hist_x_trackdydz->SetAxisRange(-10., 10., "Y");
  hist_y_trackdydz->SetAxisRange(-20., 20., "Y");
  hist_dxdz_trackdydz->SetAxisRange(-10., 10., "Y");
  hist_dydz_trackdydz->SetAxisRange(-20., 20., "Y");

  name_x += "_fit";
  name_y += "_fit";
  name_dxdz += "_fit";
  name_dydz += "_fit";
  name_alphax += "_fit";
  name_alphay += "_fit";
  name_x_trackx += "_fit";
  name_y_trackx += "_fit";
  name_dxdz_trackx += "_fit";
  name_dydz_trackx += "_fit";
  name_x_tracky += "_fit";
  name_y_tracky += "_fit";
  name_dxdz_tracky += "_fit";
  name_dydz_tracky += "_fit";
  name_x_trackdxdz += "_fit";
  name_y_trackdxdz += "_fit";
  name_dxdz_trackdxdz += "_fit";
  name_dydz_trackdxdz += "_fit";
  name_x_trackdydz += "_fit";
  name_y_trackdydz += "_fit";
  name_dxdz_trackdydz += "_fit";
  name_dydz_trackdydz += "_fit";

  TF1 *fit_x = NULL;
  TF1 *fit_y = NULL;
  TF1 *fit_dxdz = NULL;
  TF1 *fit_dydz = NULL;
  if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D) {
    fit_x = new TF1(name_x.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_x, max_x, 3);
    fit_x->SetParameters(sum_of_weights * (max_x - min_x)/bins, 10.*value(kAlignX), 10.*value(kResidXSigma));
    const double er_x[3] = {0., 10.*errorerror(kAlignX), 10.*errorerror(kResidXSigma)};
    fit_x->SetParErrors(er_x);
    fit_y = new TF1(name_y.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_y, max_y, 3);
    fit_y->SetParameters(sum_of_weights * (max_y - min_y)/bins, 10.*value(kAlignY), 10.*value(kResidYSigma));
    const double er_y[3] = {0., 10.*errorerror(kAlignY), 10.*errorerror(kResidYSigma)};
    fit_y->SetParErrors(er_y);
    fit_dxdz = new TF1(name_dxdz.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_dxdz, max_dxdz, 3);
    fit_dxdz->SetParameters(sum_of_weights * (max_dxdz - min_dxdz)/bins, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma));
    const double er_dxdz[3] = {0., 1000.*errorerror(kAlignPhiY), 1000.*errorerror(kResSlopeXSigma)};
    fit_dxdz->SetParErrors(er_dxdz);
    fit_dydz = new TF1(name_dydz.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_dydz, max_dydz, 3);
    fit_dydz->SetParameters(sum_of_weights * (max_dydz - min_dydz)/bins, -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma));
    const double er_dydz[3] = {0., 1000.*errorerror(kAlignPhiX), 1000.*errorerror(kResSlopeYSigma)};
    fit_dydz->SetParErrors(er_dydz);
  }
  else if (residualsModel() == kPowerLawTails) {
    fit_x = new TF1(name_x.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_x, max_x, 4);
    fit_x->SetParameters(sum_of_weights * (max_x - min_x)/bins, 10.*value(kAlignX), 10.*value(kResidXSigma), 10.*value(kResidXGamma));
    fit_y = new TF1(name_y.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_y, max_y, 4);
    fit_y->SetParameters(sum_of_weights * (max_y - min_y)/bins, 10.*value(kAlignY), 10.*value(kResidYSigma), 10.*value(kResidYGamma));
    fit_dxdz = new TF1(name_dxdz.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_dxdz, max_dxdz, 4);
    fit_dxdz->SetParameters(sum_of_weights * (max_dxdz - min_dxdz)/bins, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma), 1000.*value(kResSlopeXGamma));
    fit_dydz = new TF1(name_dydz.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_dydz, max_dydz, 4);
    fit_dydz->SetParameters(sum_of_weights * (max_dydz - min_dydz)/bins, -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma), 1000.*value(kResSlopeYGamma));
  }
  else if (residualsModel() == kROOTVoigt) {
    fit_x = new TF1(name_x.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_x, max_x, 4);
    fit_x->SetParameters(sum_of_weights * (max_x - min_x)/bins, 10.*value(kAlignX), 10.*value(kResidXSigma), 10.*value(kResidXGamma));
    fit_y = new TF1(name_y.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_y, max_y, 4);
    fit_y->SetParameters(sum_of_weights * (max_y - min_y)/bins, 10.*value(kAlignY), 10.*value(kResidYSigma), 10.*value(kResidYGamma));
    fit_dxdz = new TF1(name_dxdz.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_dxdz, max_dxdz, 4);
    fit_dxdz->SetParameters(sum_of_weights * (max_dxdz - min_dxdz)/bins, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma), 1000.*value(kResSlopeXGamma));
    fit_dydz = new TF1(name_dydz.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_dydz, max_dydz, 4);
    fit_dydz->SetParameters(sum_of_weights * (max_dydz - min_dydz)/bins, -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma), 1000.*value(kResSlopeYGamma));
  }
  else if (residualsModel() == kGaussPowerTails) {
    fit_x = new TF1(name_x.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_x, max_x, 3);
    fit_x->SetParameters(sum_of_weights * (max_x - min_x)/bins, 10.*value(kAlignX), 10.*value(kResidXSigma));
    fit_y = new TF1(name_y.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_y, max_y, 3);
    fit_y->SetParameters(sum_of_weights * (max_y - min_y)/bins, 10.*value(kAlignY), 10.*value(kResidYSigma));
    fit_dxdz = new TF1(name_dxdz.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_dxdz, max_dxdz, 3);
    fit_dxdz->SetParameters(sum_of_weights * (max_dxdz - min_dxdz)/bins, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma));
    fit_dydz = new TF1(name_dydz.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_dydz, max_dydz, 3);
    fit_dydz->SetParameters(sum_of_weights * (max_dydz - min_dydz)/bins, -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma));
  }
  else { assert(false); }

  fit_x->SetLineColor(2);     fit_x->SetLineWidth(2);     fit_x->Write();
  fit_y->SetLineColor(2);     fit_y->SetLineWidth(2);     fit_y->Write();
  fit_dxdz->SetLineColor(2);  fit_dxdz->SetLineWidth(2);  fit_dxdz->Write();
  fit_dydz->SetLineColor(2);  fit_dydz->SetLineWidth(2);  fit_dydz->Write();

  TF1 *fit_alphax = new TF1(name_alphax.c_str(), "[0] + x*[1]", min_dxdz, max_dxdz);
  TF1 *fit_alphay = new TF1(name_alphay.c_str(), "[0] + x*[1]", min_dydz, max_dydz);
  double aX = 10.*value(kAlignX), bX = 10.*value(kAlphaX)/1000.;
  double aY = 10.*value(kAlignY), bY = 10.*value(kAlphaY)/1000.;
  if (residualsModel() == kPureGaussian2D)
  {
    double sx = 10.*value(kResidXSigma), sy = 1000.*value(kResSlopeXSigma), r = value(kAlphaX);
    aX = mean_residualx;
    bX = 0.;
    if ( sx != 0. )
    {
      bX = 1./(sy/sx*r);
      aX = - bX * mean_resslopex;
    }
    sx = 10.*value(kResidYSigma); sy = 1000.*value(kResSlopeYSigma); r = value(kAlphaY);
    aY = mean_residualx;
    bY = 0.;
    if ( sx != 0. )
    {
      bY = 1./(sy/sx*r);
      aY = - bY * mean_resslopey;
    }
  }
  fit_alphax->SetParameters(aX, bX);
  fit_alphay->SetParameters(aY, bY);

  fit_alphax->SetLineColor(2);  fit_alphax->SetLineWidth(2);  fit_alphax->Write();
  fit_alphay->SetLineColor(2);  fit_alphay->SetLineWidth(2);  fit_alphay->Write();

  TProfile *fit_x_trackx       = dir->make<TProfile>(name_x_trackx.c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_y_trackx       = dir->make<TProfile>(name_y_trackx.c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_dxdz_trackx    = dir->make<TProfile>(name_dxdz_trackx.c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_dydz_trackx    = dir->make<TProfile>(name_dydz_trackx.c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_x_tracky       = dir->make<TProfile>(name_x_tracky.c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_y_tracky       = dir->make<TProfile>(name_y_tracky.c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_dxdz_tracky    = dir->make<TProfile>(name_dxdz_tracky.c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_dydz_tracky    = dir->make<TProfile>(name_dydz_tracky.c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_x_trackdxdz    = dir->make<TProfile>(name_x_trackdxdz.c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_y_trackdxdz    = dir->make<TProfile>(name_y_trackdxdz.c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_dxdz_trackdxdz = dir->make<TProfile>(name_dxdz_trackdxdz.c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_dydz_trackdxdz = dir->make<TProfile>(name_dydz_trackdxdz.c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_x_trackdydz    = dir->make<TProfile>(name_x_trackdydz.c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_y_trackdydz    = dir->make<TProfile>(name_y_trackdydz.c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_dxdz_trackdydz = dir->make<TProfile>(name_dxdz_trackdydz.c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_dydz_trackdydz = dir->make<TProfile>(name_dydz_trackdydz.c_str(), "", 500, min_trackdydz, max_trackdydz);

  fit_x_trackx->SetLineColor(2);        fit_x_trackx->SetLineWidth(2);
  fit_y_trackx->SetLineColor(2);        fit_y_trackx->SetLineWidth(2);
  fit_dxdz_trackx->SetLineColor(2);     fit_dxdz_trackx->SetLineWidth(2);
  fit_dydz_trackx->SetLineColor(2);     fit_dydz_trackx->SetLineWidth(2);
  fit_x_tracky->SetLineColor(2);        fit_x_tracky->SetLineWidth(2);
  fit_y_tracky->SetLineColor(2);        fit_y_tracky->SetLineWidth(2);
  fit_dxdz_tracky->SetLineColor(2);     fit_dxdz_tracky->SetLineWidth(2);
  fit_dydz_tracky->SetLineColor(2);     fit_dydz_tracky->SetLineWidth(2);
  fit_x_trackdxdz->SetLineColor(2);     fit_x_trackdxdz->SetLineWidth(2);
  fit_y_trackdxdz->SetLineColor(2);     fit_y_trackdxdz->SetLineWidth(2);
  fit_dxdz_trackdxdz->SetLineColor(2);  fit_dxdz_trackdxdz->SetLineWidth(2);
  fit_dydz_trackdxdz->SetLineColor(2);  fit_dydz_trackdxdz->SetLineWidth(2);
  fit_x_trackdydz->SetLineColor(2);     fit_x_trackdydz->SetLineWidth(2);
  fit_y_trackdydz->SetLineColor(2);     fit_y_trackdydz->SetLineWidth(2);
  fit_dxdz_trackdydz->SetLineColor(2);  fit_dxdz_trackdydz->SetLineWidth(2);
  fit_dydz_trackdydz->SetLineColor(2);  fit_dydz_trackdydz->SetLineWidth(2);

  name_x_trackx += "line";
  name_y_trackx += "line";
  name_dxdz_trackx += "line";
  name_dydz_trackx += "line";
  name_x_tracky += "line";
  name_y_tracky += "line";
  name_dxdz_tracky += "line";
  name_dydz_tracky += "line";
  name_x_trackdxdz += "line";
  name_y_trackdxdz += "line";
  name_dxdz_trackdxdz += "line";
  name_dydz_trackdxdz += "line";
  name_x_trackdydz += "line";
  name_y_trackdydz += "line";
  name_dxdz_trackdydz += "line";
  name_dydz_trackdydz += "line";

  TF1 *fitline_x_trackx       = new TF1(name_x_trackx.c_str(), residual_x_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_y_trackx       = new TF1(name_y_trackx.c_str(), residual_y_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_dxdz_trackx    = new TF1(name_dxdz_trackx.c_str(), residual_dxdz_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_dydz_trackx    = new TF1(name_dydz_trackx.c_str(), residual_dydz_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_x_tracky       = new TF1(name_x_tracky.c_str(), residual_x_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_y_tracky       = new TF1(name_y_tracky.c_str(), residual_y_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_dxdz_tracky    = new TF1(name_dxdz_tracky.c_str(), residual_dxdz_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_dydz_tracky    = new TF1(name_dydz_tracky.c_str(), residual_dydz_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_x_trackdxdz    = new TF1(name_x_trackdxdz.c_str(), residual_x_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_y_trackdxdz    = new TF1(name_y_trackdxdz.c_str(), residual_y_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_dxdz_trackdxdz = new TF1(name_dxdz_trackdxdz.c_str(), residual_dxdz_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_dydz_trackdxdz = new TF1(name_dydz_trackdxdz.c_str(), residual_dydz_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_x_trackdydz    = new TF1(name_x_trackdydz.c_str(), residual_x_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_y_trackdydz    = new TF1(name_y_trackdydz.c_str(), residual_y_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_dxdz_trackdydz = new TF1(name_dxdz_trackdydz.c_str(), residual_dxdz_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_dydz_trackdydz = new TF1(name_dydz_trackdydz.c_str(), residual_dydz_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);

  std::vector<TF1*> fitlines;
  fitlines.push_back(fitline_x_trackx);
  fitlines.push_back(fitline_y_trackx);
  fitlines.push_back(fitline_dxdz_trackx);
  fitlines.push_back(fitline_dydz_trackx);
  fitlines.push_back(fitline_x_tracky);
  fitlines.push_back(fitline_y_tracky);
  fitlines.push_back(fitline_dxdz_tracky);
  fitlines.push_back(fitline_dydz_tracky);
  fitlines.push_back(fitline_x_trackdxdz);
  fitlines.push_back(fitline_y_trackdxdz);
  fitlines.push_back(fitline_dxdz_trackdxdz);
  fitlines.push_back(fitline_dydz_trackdxdz);
  fitlines.push_back(fitline_x_trackdydz);
  fitlines.push_back(fitline_y_trackdydz);
  fitlines.push_back(fitline_dxdz_trackdydz);
  fitlines.push_back(fitline_dydz_trackdydz);

  double fitparameters[14] = {value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ),
                              mean_trackx, mean_tracky, mean_trackdxdz, mean_trackdydz,
                              value(kAlphaX), mean_resslopex, value(kAlphaY), mean_resslopey};
  if (residualsModel() == kPureGaussian2D) fitparameters[10] = fitparameters[12] = 0.;

  for(std::vector<TF1*>::const_iterator itr = fitlines.begin(); itr != fitlines.end(); itr++)
  {
    (*itr)->SetParameters(fitparameters);
    (*itr)->SetLineColor(2);
    (*itr)->SetLineWidth(2);
    (*itr)->Write();
  }

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter)
  {
    const double residX    = (*resiter)[kResidX];
    const double residY    = (*resiter)[kResidY];
    const double resslopeX = (*resiter)[kResSlopeX];
    const double resslopeY = (*resiter)[kResSlopeY];
    const double positionX = (*resiter)[kPositionX];
    const double positionY = (*resiter)[kPositionY];
    const double angleX    = (*resiter)[kAngleX];
    const double angleY    = (*resiter)[kAngleY];
    const double redchi2   = (*resiter)[kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed

      hist_alphax->Fill(1000.*resslopeX, 10.*residX);
      hist_alphay->Fill(1000.*resslopeY, 10.*residY);

      double coefX = value(kAlphaX), coefY = value(kAlphaY);
      if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D) coefX = coefY = 0.;
      double geom_residX = residual_x(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, coefX, resslopeX);
      hist_x->Fill(10.*(residX - geom_residX + value(kAlignX)), weight);
      hist_x_trackx->Fill(positionX, 10.*residX, weight);
      hist_x_tracky->Fill(positionY, 10.*residX, weight);
      hist_x_trackdxdz->Fill(angleX, 10.*residX, weight);
      hist_x_trackdydz->Fill(angleY, 10.*residX, weight);
      fit_x_trackx->Fill(positionX, 10.*geom_residX, weight);
      fit_x_tracky->Fill(positionY, 10.*geom_residX, weight);
      fit_x_trackdxdz->Fill(angleX, 10.*geom_residX, weight);
      fit_x_trackdydz->Fill(angleY, 10.*geom_residX, weight);

      double geom_residY = residual_y(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, coefY, resslopeY);
      hist_y->Fill(10.*(residY - geom_residY + value(kAlignY)), weight);
      hist_y_trackx->Fill(positionX, 10.*residY, weight);
      hist_y_tracky->Fill(positionY, 10.*residY, weight);
      hist_y_trackdxdz->Fill(angleX, 10.*residY, weight);
      hist_y_trackdydz->Fill(angleY, 10.*residY, weight);
      fit_y_trackx->Fill(positionX, 10.*geom_residY, weight);
      fit_y_tracky->Fill(positionY, 10.*geom_residY, weight);
      fit_y_trackdxdz->Fill(angleX, 10.*geom_residY, weight);
      fit_y_trackdydz->Fill(angleY, 10.*geom_residY, weight);

      double geom_resslopeX = residual_dxdz(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY);
      hist_dxdz->Fill(1000.*(resslopeX - geom_resslopeX + value(kAlignPhiY)), weight);
      hist_dxdz_trackx->Fill(positionX, 1000.*resslopeX, weight);
      hist_dxdz_tracky->Fill(positionY, 1000.*resslopeX, weight);
      hist_dxdz_trackdxdz->Fill(angleX, 1000.*resslopeX, weight);
      hist_dxdz_trackdydz->Fill(angleY, 1000.*resslopeX, weight);
      fit_dxdz_trackx->Fill(positionX, 1000.*geom_resslopeX, weight);
      fit_dxdz_tracky->Fill(positionY, 1000.*geom_resslopeX, weight);
      fit_dxdz_trackdxdz->Fill(angleX, 1000.*geom_resslopeX, weight);
      fit_dxdz_trackdydz->Fill(angleY, 1000.*geom_resslopeX, weight);

      double geom_resslopeY = residual_dydz(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY);
      hist_dydz->Fill(1000.*(resslopeY - geom_resslopeY - value(kAlignPhiX)), weight);
      hist_dydz_trackx->Fill(positionX, 1000.*resslopeY, weight);
      hist_dydz_tracky->Fill(positionY, 1000.*resslopeY, weight);
      hist_dydz_trackdxdz->Fill(angleX, 1000.*resslopeY, weight);
      hist_dydz_trackdydz->Fill(angleY, 1000.*resslopeY, weight);
      fit_dydz_trackx->Fill(positionX, 1000.*geom_resslopeY, weight);
      fit_dydz_tracky->Fill(positionY, 1000.*geom_resslopeY, weight);
      fit_dydz_trackdxdz->Fill(angleX, 1000.*geom_resslopeY, weight);
      fit_dydz_trackdydz->Fill(angleY, 1000.*geom_resslopeY, weight);
    }

    hist_x_raw->Fill(10.*residX);
    hist_y_raw->Fill(10.*residY);
    hist_dxdz_raw->Fill(1000.*resslopeX);
    hist_dydz_raw->Fill(1000.*resslopeY);
    if (fabs(resslopeX) < 0.005) hist_x_cut->Fill(10.*residX);
    if (fabs(resslopeY) < 0.030) hist_y_cut->Fill(10.*residY);
  }

  double chi2 = 0.;
  double ndof = 0.;
  for (int i = 1;  i <= hist_x->GetNbinsX();  i++) {
    double xi = hist_x->GetBinCenter(i);
    double yi = hist_x->GetBinContent(i);
    double yerri = hist_x->GetBinError(i);
    double yth = fit_x->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  for (int i = 1;  i <= hist_y->GetNbinsX();  i++) {
    double xi = hist_y->GetBinCenter(i);
    double yi = hist_y->GetBinContent(i);
    double yerri = hist_y->GetBinError(i);
    double yth = fit_y->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  for (int i = 1;  i <= hist_dxdz->GetNbinsX();  i++) {
    double xi = hist_dxdz->GetBinCenter(i);
    double yi = hist_dxdz->GetBinContent(i);
    double yerri = hist_dxdz->GetBinError(i);
    double yth = fit_dxdz->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  for (int i = 1;  i <= hist_dydz->GetNbinsX();  i++) {
    double xi = hist_dydz->GetBinCenter(i);
    double yi = hist_dydz->GetBinContent(i);
    double yerri = hist_dydz->GetBinError(i);
    double yth = fit_dydz->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  ndof -= npar();

  return (ndof > 0. ? chi2 / ndof : -1.);
}


TTree * MuonResiduals6DOFFitter::readNtuple(std::string fname, unsigned int wheel, unsigned int station, unsigned int sector, unsigned int preselected)
{
  TFile *f = new TFile(fname.c_str());
  TTree *t = (TTree*)f->Get("mual_ttree");

  // Create  new temporary file
  TFile *tmpf = new TFile("small_tree_fit.root","recreate");
  assert(tmpf!=0);

  // filter the tree (temporarily lives in the new file)
  TTree *tt = t->CopyTree(Form("is_dt && ring_wheel==%d && station==%d && sector==%d && select==%d", wheel, station, sector, (bool)preselected));

  MuonAlignmentTreeRow r;
  tt->SetBranchAddress("res_x", &r.res_x);
  tt->SetBranchAddress("res_slope_x", &r.res_slope_x);
  tt->SetBranchAddress("res_y", &r.res_y);
  tt->SetBranchAddress("res_slope_y", &r.res_slope_y);
  tt->SetBranchAddress("pos_x", &r.pos_x);
  tt->SetBranchAddress("pos_y", &r.pos_y);
  tt->SetBranchAddress("angle_x", &r.angle_x);
  tt->SetBranchAddress("angle_y", &r.angle_y);
  tt->SetBranchAddress("pz", &r.pz);
  tt->SetBranchAddress("pt", &r.pt);
  tt->SetBranchAddress("q", &r.q);

  Long64_t nentries = tt->GetEntries();
  for (Long64_t i=0;i<nentries; i++)
  {
    tt->GetEntry(i);
    double *rdata = new double[MuonResiduals6DOFFitter::kNData];
    rdata[kResidX] = r.res_x;
    rdata[kResidY] = r.res_y;
    rdata[kResSlopeX] = r.res_slope_x;
    rdata[kResSlopeY] = r.res_slope_y;
    rdata[kPositionX] = r.pos_x;
    rdata[kPositionY] = r.pos_y;
    rdata[kAngleX] = r.angle_x;
    rdata[kAngleY] = r.angle_y;
    rdata[kRedChi2] = 0.1;
    rdata[kPz] = r.pz;
    rdata[kPt] = r.pt;
    rdata[kCharge] = r.q;
    fill(rdata);
  }
  delete f;
  //delete tmpf;
  return tt;
}
