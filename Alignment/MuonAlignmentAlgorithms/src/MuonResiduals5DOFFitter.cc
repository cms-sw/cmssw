// $Id: MuonResiduals5DOFFitter.cc,v 1.9 2011/10/12 23:44:11 khotilov Exp $

#ifdef STANDALONE_FITTER
#include "MuonResiduals5DOFFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals5DOFFitter.h"
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

  double residual_x(double delta_x, double delta_z,
                    double delta_phix, double delta_phiy, double delta_phiz,
                    double track_x, double track_y, double track_dxdz, double track_dydz,
                    double alpha, double resslope)
  {
    return delta_x
        - track_dxdz * delta_z
        - track_y * track_dxdz * delta_phix
        + track_x * track_dxdz * delta_phiy
        - track_y * delta_phiz
        + resslope * alpha;
  }

  double residual_dxdz(double delta_x, double delta_z,
                       double delta_phix, double delta_phiy, double delta_phiz,
                       double track_x, double track_y, double track_dxdz, double track_dydz)
  {
    return -track_dxdz * track_dydz * delta_phix
        + (1. + track_dxdz * track_dxdz) * delta_phiy
        - track_dydz * delta_phiz;
  }

  Double_t residual_x_trackx_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[2], p[3], p[4], p[5], xx[0], p[7], p[8], p[9], p[10], p[11]); }
  Double_t residual_x_tracky_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[2], p[3], p[4], p[5], p[6], xx[0], p[8], p[9], p[10], p[11]); }
  Double_t residual_x_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[2], p[3], p[4], p[5], p[6], p[7], xx[0], p[9], p[10], p[11]); }
  Double_t residual_x_trackdydz_TF1(Double_t *xx, Double_t *p) { return 10.*residual_x(p[0], p[2], p[3], p[4], p[5], p[6], p[7], p[8], xx[0], p[10], p[11]); }

  Double_t residual_dxdz_trackx_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[2], p[3], p[4], p[5], xx[0], p[7], p[8], p[9]); }
  Double_t residual_dxdz_tracky_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[2], p[3], p[4], p[5], p[6], xx[0], p[8], p[9]); }
  Double_t residual_dxdz_trackdxdz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[2], p[3], p[4], p[5], p[6], p[7], xx[0], p[9]); }
  Double_t residual_dxdz_trackdydz_TF1(Double_t *xx, Double_t *p) { return 1000.*residual_dxdz(p[0], p[2], p[3], p[4], p[5], p[6], p[7], p[8], xx[0]); }
}


void MuonResiduals5DOFFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag)
{
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(minuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter)
  {
    const double residual = (*resiter)[MuonResiduals5DOFFitter::kResid];
    const double resslope = (*resiter)[MuonResiduals5DOFFitter::kResSlope];
    const double positionX = (*resiter)[MuonResiduals5DOFFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals5DOFFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals5DOFFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals5DOFFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals5DOFFitter::kRedChi2];

    const double alignx = par[MuonResiduals5DOFFitter::kAlignX];
    const double alignz = par[MuonResiduals5DOFFitter::kAlignZ];
    const double alignphix = par[MuonResiduals5DOFFitter::kAlignPhiX];
    const double alignphiy = par[MuonResiduals5DOFFitter::kAlignPhiY];
    const double alignphiz = par[MuonResiduals5DOFFitter::kAlignPhiZ];
    const double residsigma = par[MuonResiduals5DOFFitter::kResidSigma];
    const double resslopesigma = par[MuonResiduals5DOFFitter::kResSlopeSigma];
    const double alpha = par[MuonResiduals5DOFFitter::kAlpha];
    const double residgamma = par[MuonResiduals5DOFFitter::kResidGamma];
    const double resslopegamma = par[MuonResiduals5DOFFitter::kResSlopeGamma];

    double coeff = alpha;
    if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian ||
        fitter->residualsModel() == MuonResidualsFitter::kPureGaussian2D) coeff = 0.;
    double residpeak = residual_x(alignx, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, coeff, resslope);
    double resslopepeak = residual_dxdz(alignx, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY);

    double weight = (1./redchi2) * number_of_hits / sum_of_weights;
    if (!weight_alignment) weight = 1.;

    if (!weight_alignment  ||  TMath::Prob(redchi2*8, 8) < 0.99)  // no spikes allowed
    {
      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
        if (fitter->useRes() == MuonResidualsFitter::k1111 || fitter->useRes() == MuonResidualsFitter::k1110 || fitter->useRes() == MuonResidualsFitter::k1010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residual, residpeak, residsigma);
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslope, resslopepeak, resslopesigma);
	}
        else if (fitter->useRes() == MuonResidualsFitter::k1100) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residual, residpeak, residsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k0010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslope, resslopepeak, resslopesigma);
        }
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian2D) {
        if (fitter->useRes() == MuonResidualsFitter::k1111 || fitter->useRes() == MuonResidualsFitter::k1110 || fitter->useRes() == MuonResidualsFitter::k1010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian2D(residual, resslope, residpeak, resslopepeak, residsigma, resslopesigma, alpha);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k1100) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(residual, residpeak, residsigma);
        }
        else if (fitter->useRes() == MuonResidualsFitter::k0010) {
          fval += -weight * MuonResidualsFitter_logPureGaussian(resslope, resslopepeak, resslopesigma);
        }
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
        fval += -weight * MuonResidualsFitter_logPowerLawTails(residual, residpeak, residsigma, residgamma);
        fval += -weight * MuonResidualsFitter_logPowerLawTails(resslope, resslopepeak, resslopesigma, resslopegamma);
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
        fval += -weight * MuonResidualsFitter_logROOTVoigt(residual, residpeak, residsigma, residgamma);
        fval += -weight * MuonResidualsFitter_logROOTVoigt(resslope, resslopepeak, resslopesigma, resslopegamma);
      }
      else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(residual, residpeak, residsigma);
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(resslope, resslopepeak, resslopesigma);
      }
      else { assert(false); }
    }
  }
}


void MuonResiduals5DOFFitter::correctBField()
{
  MuonResidualsFitter::correctBField(kPt, kCharge);
}


void MuonResiduals5DOFFitter::inform(TMinuit *tMinuit)
{
  minuit = tMinuit;
}


double MuonResiduals5DOFFitter::sumofweights()
{
  sum_of_weights = 0.;
  number_of_hits = 0.;
  weight_alignment = m_weightAlignment;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    if (m_weightAlignment) {
      double redchi2 = (*resiter)[MuonResiduals5DOFFitter::kRedChi2];
      if (TMath::Prob(redchi2*8, 8) < 0.99) {
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


bool MuonResiduals5DOFFitter::fit(Alignable *ali)
{
  initialize_table();  // if not already initialized
  sumofweights();

  double res_std = 0.5;
  double resslope_std = 0.005;

  int nums[10]          = {kAlignX, kAlignZ, kAlignPhiX, kAlignPhiY, kAlignPhiZ,    kResidSigma, kResSlopeSigma,   kAlpha,    kResidGamma, kResSlopeGamma};
  std::string names[10] = {"AlignX","AlignZ","AlignPhiX","AlignPhiY","AlignPhiZ",   "ResidSigma","ResSlopeSigma",  "Alpha",   "ResidGamma","ResSlopeGamma"};
  double starts[10]     = {0., 0., 0., 0., 0.,              res_std, resslope_std,               0.,     0.1*res_std, 0.1*resslope_std};
  double steps[10]      = {0.1, 0.1, 0.001, 0.001, 0.001,   0.001*res_std, 0.001*resslope_std,   0.001,  0.01*res_std, 0.01*resslope_std};
  double lows[10]       = {0., 0., 0., 0., 0.,    0., 0.,      -1.,   0., 0.};
  double highs[10]      = {0., 0., 0., 0., 0.,    10., 0.1,     1.,   0., 0.};

  std::vector<int> num(nums, nums+5);
  std::vector<std::string> name(names, names+5);
  std::vector<double> start(starts, starts+5);
  std::vector<double> step(steps, steps+5);
  std::vector<double> low(lows, lows+5);
  std::vector<double> high(highs, highs+5);

  bool add_alpha = ( residualsModel() == kPureGaussian2D);
  bool add_gamma = ( residualsModel() == kROOTVoigt || residualsModel() == kPowerLawTails);

  int idx[4], ni = 0;
  if (useRes() == k1111 || useRes() == k1110 || useRes() == k1010) {
    for(ni=0; ni<2; ni++) idx[ni] = ni+5;
    if (add_alpha) idx[ni++] = 7;
    else if (add_gamma) for(; ni<4; ni++) idx[ni] = ni+6;
    if (!add_alpha) fix(kAlpha);
  }
  else if (useRes() == k1100) {
    idx[ni++] = 5;
    if (add_gamma) idx[ni++] = 8;
    fix(kResSlopeSigma);
    fix(kAlpha);
  }
  else if (useRes() == k0010) {
    idx[ni++] = 6;
    if (add_gamma) idx[ni++] = 9;
    fix(kResidSigma);
    fix(kAlpha);
  }
  for (int i=0; i<ni; i++){
    num.push_back(nums[idx[i]]);
    name.push_back(names[idx[i]]);
    start.push_back(starts[idx[i]]);
    step.push_back(steps[idx[i]]);
    low.push_back(lows[idx[i]]);
    high.push_back(highs[idx[i]]);
  }

  return dofit(&MuonResiduals5DOFFitter_FCN, num, name, start, step, low, high);
}


double MuonResiduals5DOFFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali)
{
  sumofweights();

  double mean_residual = 0., mean_resslope = 0.;
  double mean_trackx = 0., mean_tracky = 0., mean_trackdxdz = 0., mean_trackdydz = 0.;
  double sum_w = 0.;

  for (std::vector<double*>::const_iterator rit = residuals_begin();  rit != residuals_end();  ++rit)
  {
    const double redchi2 = (*rit)[kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*6, 6) < 0.99)  // no spikes allowed
    {
      double factor_w = 1./(sum_w + weight);
      mean_residual  = factor_w * (sum_w * mean_residual  + weight * (*rit)[kResid]);
      mean_resslope  = factor_w * (sum_w * mean_resslope  + weight * (*rit)[kResSlope]);
      mean_trackx    = factor_w * (sum_w * mean_trackx    + weight * (*rit)[kPositionX]);
      mean_tracky    = factor_w * (sum_w * mean_tracky    + weight * (*rit)[kPositionY]);
      mean_trackdxdz = factor_w * (sum_w * mean_trackdxdz + weight * (*rit)[kAngleX]);
      mean_trackdydz = factor_w * (sum_w * mean_trackdydz + weight * (*rit)[kAngleY]);
      sum_w += weight;
    }
  }

  std::string name_residual, name_resslope, name_residual_raw, name_resslope_raw, name_residual_cut, name_alpha;
  std::string name_residual_trackx, name_resslope_trackx;
  std::string name_residual_tracky, name_resslope_tracky;
  std::string name_residual_trackdxdz, name_resslope_trackdxdz;
  std::string name_residual_trackdydz, name_resslope_trackdydz;

  name_residual = name + "_residual";
  name_resslope = name + "_resslope";
  name_residual_raw = name + "_residual_raw";
  name_resslope_raw = name + "_resslope_raw";
  name_residual_cut = name + "_residual_cut";
  name_alpha = name + "_alpha";
  name_residual_trackx = name + "_residual_trackx";
  name_resslope_trackx = name + "_resslope_trackx";
  name_residual_tracky = name + "_residual_tracky";
  name_resslope_tracky = name + "_resslope_tracky";
  name_residual_trackdxdz = name + "_residual_trackdxdz";
  name_resslope_trackdxdz = name + "_resslope_trackdxdz";
  name_residual_trackdydz = name + "_residual_trackdydz";
  name_resslope_trackdydz = name + "_resslope_trackdydz";

  double width = ali->surface().width();
  double length = ali->surface().length();
  int    bins_residual = 150, bins_resslope = 100;
  double min_residual = -75.,    max_residual = 75.;
  double min_resslope = -50.,    max_resslope = 50.;
  double min_trackx = -width/2.,  max_trackx = width/2.;
  double min_tracky = -length/2., max_tracky = length/2.;
  double min_trackdxdz = -1.5,    max_trackdxdz = 1.5;
  double min_trackdydz = -1.5,    max_trackdydz = 1.5;

  TH1F *hist_residual = dir->make<TH1F>(name_residual.c_str(), "", bins_residual, min_residual, max_residual);
  TH1F *hist_resslope = dir->make<TH1F>(name_resslope.c_str(), "", bins_resslope, min_resslope, max_resslope);
  TH1F *hist_residual_raw = dir->make<TH1F>(name_residual_raw.c_str(), "", bins_residual, min_residual, max_residual);
  TH1F *hist_resslope_raw = dir->make<TH1F>(name_resslope_raw.c_str(), "", bins_resslope, min_resslope, max_resslope);
  TH1F *hist_residual_cut = dir->make<TH1F>(name_residual_cut.c_str(), "", bins_residual, min_residual, max_residual);
  TH2F *hist_alpha = dir->make<TH2F>(name_alpha.c_str(), "", 50, min_resslope, max_resslope, 50, -50., 50.);

  TProfile *hist_residual_trackx = dir->make<TProfile>(name_residual_trackx.c_str(), "", 50, min_trackx, max_trackx, min_residual, max_residual);
  TProfile *hist_resslope_trackx = dir->make<TProfile>(name_resslope_trackx.c_str(), "", 50, min_trackx, max_trackx, min_resslope, max_resslope);
  TProfile *hist_residual_tracky = dir->make<TProfile>(name_residual_tracky.c_str(), "", 50, min_tracky, max_tracky, min_residual, max_residual);
  TProfile *hist_resslope_tracky = dir->make<TProfile>(name_resslope_tracky.c_str(), "", 50, min_tracky, max_tracky, min_resslope, max_resslope);
  TProfile *hist_residual_trackdxdz = dir->make<TProfile>(name_residual_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_residual, max_residual);
  TProfile *hist_resslope_trackdxdz = dir->make<TProfile>(name_resslope_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz, min_resslope, max_resslope);
  TProfile *hist_residual_trackdydz = dir->make<TProfile>(name_residual_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_residual, max_residual);
  TProfile *hist_resslope_trackdydz = dir->make<TProfile>(name_resslope_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz, min_resslope, max_resslope);

  hist_residual_trackx->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackx->SetAxisRange(-10., 10., "Y");
  hist_residual_tracky->SetAxisRange(-10., 10., "Y");
  hist_resslope_tracky->SetAxisRange(-10., 10., "Y");
  hist_residual_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_residual_trackdydz->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackdydz->SetAxisRange(-10., 10., "Y");

  name_residual += "_fit";
  name_resslope += "_fit";
  name_alpha += "_fit";
  name_residual_trackx += "_fit";
  name_resslope_trackx += "_fit";
  name_residual_tracky += "_fit";
  name_resslope_tracky += "_fit";
  name_residual_trackdxdz += "_fit";
  name_resslope_trackdxdz += "_fit";
  name_residual_trackdydz += "_fit";
  name_resslope_trackdydz += "_fit";

  TF1 *fit_residual = NULL;
  TF1 *fit_resslope = NULL;
  if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D) {
    fit_residual = new TF1(name_residual.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(sum_of_weights * (max_residual - min_residual)/bins_residual, 10.*value(kAlignX), 10.*value(kResidSigma));
    const double er_res[3] = {0., 10.*errorerror(kAlignX), 10.*errorerror(kResidSigma)};
    fit_residual->SetParErrors(er_res);
    fit_resslope = new TF1(name_resslope.c_str(), MuonResidualsFitter_pureGaussian_TF1, min_resslope, max_resslope, 3);
    fit_resslope->SetParameters(sum_of_weights * (max_resslope - min_resslope)/bins_resslope, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma));
    const double er_resslope[3] = {0., 1000.*errorerror(kAlignPhiY), 1000.*errorerror(kResSlopeSigma)};
    fit_resslope->SetParErrors(er_resslope);
  }
  else if (residualsModel() == kPowerLawTails) {
    fit_residual = new TF1(name_residual.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(sum_of_weights * (max_residual - min_residual)/bins_residual, 10.*value(kAlignX), 10.*value(kResidSigma), 10.*value(kResidGamma));
    fit_resslope = new TF1(name_resslope.c_str(), MuonResidualsFitter_powerLawTails_TF1, min_resslope, max_resslope, 4);
    fit_resslope->SetParameters(sum_of_weights * (max_resslope - min_resslope)/bins_resslope, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma), 1000.*value(kResSlopeGamma));
  }
  else if (residualsModel() == kROOTVoigt) {
    fit_residual = new TF1(name_residual.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(sum_of_weights * (max_residual - min_residual)/bins_residual, 10.*value(kAlignX), 10.*value(kResidSigma), 10.*value(kResidGamma));
    fit_resslope = new TF1(name_resslope.c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_resslope, max_resslope, 4);
    fit_resslope->SetParameters(sum_of_weights * (max_resslope - min_resslope)/bins_resslope, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma), 1000.*value(kResSlopeGamma));
  }
  else if (residualsModel() == kGaussPowerTails) {
    fit_residual = new TF1(name_residual.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(sum_of_weights * (max_residual - min_residual)/bins_residual, 10.*value(kAlignX), 10.*value(kResidSigma));
    fit_resslope = new TF1(name_resslope.c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_resslope, max_resslope, 3);
    fit_resslope->SetParameters(sum_of_weights * (max_resslope - min_resslope)/bins_resslope, 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma));
  }
  else { assert(false); }

  fit_residual->SetLineColor(2);  fit_residual->SetLineWidth(2);  fit_residual->Write();
  fit_resslope->SetLineColor(2);  fit_resslope->SetLineWidth(2);  fit_resslope->Write();

  TF1 *fit_alpha = new TF1(name_alpha.c_str(), "[0] + x*[1]", min_resslope, max_resslope);
  double a = 10.*value(kAlignX), b = 10.*value(kAlpha)/1000.;
  if (residualsModel() == kPureGaussian2D)
  {
    double sx = 10.*value(kResidSigma), sy = 1000.*value(kResSlopeSigma), r = value(kAlpha);
    a = mean_residual;
    b = 0.;
    if ( sx != 0. )
    {
      b = 1./(sy/sx*r);
      a = - b * mean_resslope;
    }
  }
  fit_alpha->SetParameters(a, b);
  fit_alpha->SetLineColor(2);  fit_alpha->SetLineWidth(2);  fit_alpha->Write();

  TProfile *fit_residual_trackx = dir->make<TProfile>(name_residual_trackx.c_str(), "", 50, min_trackx, max_trackx);
  TProfile *fit_resslope_trackx = dir->make<TProfile>(name_resslope_trackx.c_str(), "", 50, min_trackx, max_trackx);
  TProfile *fit_residual_tracky = dir->make<TProfile>(name_residual_tracky.c_str(), "", 50, min_tracky, max_tracky);
  TProfile *fit_resslope_tracky = dir->make<TProfile>(name_resslope_tracky.c_str(), "", 50, min_tracky, max_tracky);
  TProfile *fit_residual_trackdxdz = dir->make<TProfile>(name_residual_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz);
  TProfile *fit_resslope_trackdxdz = dir->make<TProfile>(name_resslope_trackdxdz.c_str(), "", 250, min_trackdxdz, max_trackdxdz);
  TProfile *fit_residual_trackdydz = dir->make<TProfile>(name_residual_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz);
  TProfile *fit_resslope_trackdydz = dir->make<TProfile>(name_resslope_trackdydz.c_str(), "", 250, min_trackdydz, max_trackdydz);

  fit_residual_trackx->SetLineColor(2);     fit_residual_trackx->SetLineWidth(2);
  fit_resslope_trackx->SetLineColor(2);     fit_resslope_trackx->SetLineWidth(2);
  fit_residual_tracky->SetLineColor(2);     fit_residual_tracky->SetLineWidth(2);
  fit_resslope_tracky->SetLineColor(2);     fit_resslope_tracky->SetLineWidth(2);
  fit_residual_trackdxdz->SetLineColor(2);  fit_residual_trackdxdz->SetLineWidth(2);
  fit_resslope_trackdxdz->SetLineColor(2);  fit_resslope_trackdxdz->SetLineWidth(2);
  fit_residual_trackdydz->SetLineColor(2);  fit_residual_trackdydz->SetLineWidth(2);
  fit_resslope_trackdydz->SetLineColor(2);  fit_resslope_trackdydz->SetLineWidth(2);

  name_residual_trackx += "line";
  name_resslope_trackx += "line";
  name_residual_tracky += "line";
  name_resslope_tracky += "line";
  name_residual_trackdxdz += "line";
  name_resslope_trackdxdz += "line";
  name_residual_trackdydz += "line";
  name_resslope_trackdydz += "line";

  TF1 *fitline_residual_trackx = new TF1(name_residual_trackx.c_str(), residual_x_trackx_TF1, min_trackx, max_trackx, 12);
  TF1 *fitline_resslope_trackx = new TF1(name_resslope_trackx.c_str(), residual_dxdz_trackx_TF1, min_trackx, max_trackx, 12);
  TF1 *fitline_residual_tracky = new TF1(name_residual_tracky.c_str(), residual_x_tracky_TF1, min_tracky, max_tracky, 12);
  TF1 *fitline_resslope_tracky = new TF1(name_resslope_tracky.c_str(), residual_dxdz_tracky_TF1, min_tracky, max_tracky, 12);
  TF1 *fitline_residual_trackdxdz = new TF1(name_residual_trackdxdz.c_str(), residual_x_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 12);
  TF1 *fitline_resslope_trackdxdz = new TF1(name_resslope_trackdxdz.c_str(), residual_dxdz_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 12);
  TF1 *fitline_residual_trackdydz = new TF1(name_residual_trackdydz.c_str(), residual_x_trackdydz_TF1, min_trackdydz, max_trackdydz, 12);
  TF1 *fitline_resslope_trackdydz = new TF1(name_resslope_trackdydz.c_str(), residual_dxdz_trackdydz_TF1, min_trackdydz, max_trackdydz, 12);

  std::vector<TF1*> fitlines;
  fitlines.push_back(fitline_residual_trackx);
  fitlines.push_back(fitline_resslope_trackx);
  fitlines.push_back(fitline_residual_tracky);
  fitlines.push_back(fitline_resslope_tracky);
  fitlines.push_back(fitline_residual_trackdxdz);
  fitlines.push_back(fitline_resslope_trackdxdz);
  fitlines.push_back(fitline_residual_trackdydz);
  fitlines.push_back(fitline_resslope_trackdydz);

  double fitparameters[12] = {value(kAlignX), 0., value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ),
                              mean_trackx, mean_tracky, mean_trackdxdz, mean_trackdydz, value(kAlpha), mean_resslope};
  if (residualsModel() == kPureGaussian2D) fitparameters[10] = 0.;

  for(std::vector<TF1*>::const_iterator itr = fitlines.begin(); itr != fitlines.end(); itr++)
  {
    (*itr)->SetParameters(fitparameters);
    (*itr)->SetLineColor(2);
    (*itr)->SetLineWidth(2);
    (*itr)->Write();
  }

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double resid = (*resiter)[kResid];
    const double resslope = (*resiter)[kResSlope];
    const double positionX = (*resiter)[kPositionX];
    const double positionY = (*resiter)[kPositionY];
    const double angleX = (*resiter)[kAngleX];
    const double angleY = (*resiter)[kAngleY];
    const double redchi2 = (*resiter)[kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*8, 8) < 0.99) {  // no spikes allowed
      hist_alpha->Fill(1000.*resslope, 10.*resid);

      double coeff = value(kAlpha);
      if (residualsModel() == kPureGaussian || residualsModel() == kPureGaussian2D) coeff = 0.;
      double geom_resid = residual_x(value(kAlignX), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, coeff, resslope);
      hist_residual->Fill(10.*(resid - geom_resid + value(kAlignX)), weight);
      hist_residual_trackx->Fill(positionX, 10.*resid, weight);
      hist_residual_tracky->Fill(positionY, 10.*resid, weight);
      hist_residual_trackdxdz->Fill(angleX, 10.*resid, weight);
      hist_residual_trackdydz->Fill(angleY, 10.*resid, weight);
      fit_residual_trackx->Fill(positionX, 10.*geom_resid, weight);
      fit_residual_tracky->Fill(positionY, 10.*geom_resid, weight);
      fit_residual_trackdxdz->Fill(angleX, 10.*geom_resid, weight);
      fit_residual_trackdydz->Fill(angleY, 10.*geom_resid, weight);

      double geom_resslope = residual_dxdz(value(kAlignX), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY);
      hist_resslope->Fill(1000.*(resslope - geom_resslope + value(kAlignPhiY)), weight);
      hist_resslope_trackx->Fill(positionX, 1000.*resslope, weight);
      hist_resslope_tracky->Fill(positionY, 1000.*resslope, weight);
      hist_resslope_trackdxdz->Fill(angleX, 1000.*resslope, weight);
      hist_resslope_trackdydz->Fill(angleY, 1000.*resslope, weight);
      fit_resslope_trackx->Fill(positionX, 1000.*geom_resslope, weight);
      fit_resslope_tracky->Fill(positionY, 1000.*geom_resslope, weight);
      fit_resslope_trackdxdz->Fill(angleX, 1000.*geom_resslope, weight);
      fit_resslope_trackdydz->Fill(angleY, 1000.*geom_resslope, weight);
    }

    hist_residual_raw->Fill(10.*resid);
    hist_resslope_raw->Fill(1000.*resslope);
    if (fabs(resslope) < 0.005) hist_residual_cut->Fill(10.*resid);
  }

  double chi2 = 0.;
  double ndof = 0.;
  for (int i = 1;  i <= hist_residual->GetNbinsX();  i++) {
    double xi = hist_residual->GetBinCenter(i);
    double yi = hist_residual->GetBinContent(i);
    double yerri = hist_residual->GetBinError(i);
    double yth = fit_residual->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  for (int i = 1;  i <= hist_resslope->GetNbinsX();  i++) {
    double xi = hist_resslope->GetBinCenter(i);
    double yi = hist_resslope->GetBinContent(i);
    double yerri = hist_resslope->GetBinError(i);
    double yth = fit_resslope->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi)/yerri, 2);
      ndof += 1.;
    }
  }
  ndof -= npar();

  return (ndof > 0. ? chi2 / ndof : -1.);
}


TTree * MuonResiduals5DOFFitter::readNtuple(std::string fname, unsigned int wheel, unsigned int station, unsigned int sector, unsigned int preselected)
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
    double *rdata = new double[MuonResiduals5DOFFitter::kNData];
    rdata[kResid] = r.res_x;
    rdata[kResSlope] = r.res_slope_x;
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
