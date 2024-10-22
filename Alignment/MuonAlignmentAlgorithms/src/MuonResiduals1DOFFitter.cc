#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals1DOFFitter.h"
#include "TMath.h"

static TMinuit *MuonResiduals1DOFFitter_TMinuit;
static double MuonResiduals1DOFFitter_sum_of_weights;
static double MuonResiduals1DOFFitter_number_of_hits;
static bool MuonResiduals1DOFFitter_weightAlignment;

void MuonResiduals1DOFFitter::inform(TMinuit *tMinuit) { MuonResiduals1DOFFitter_TMinuit = tMinuit; }

void MuonResiduals1DOFFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo *)(MuonResiduals1DOFFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double *>::const_iterator resiter = fitter->residuals_begin(); resiter != fitter->residuals_end();
       ++resiter) {
    const double residual = (*resiter)[MuonResiduals1DOFFitter::kResid];
    const double redchi2 = (*resiter)[MuonResiduals1DOFFitter::kRedChi2];

    const double residpeak = par[MuonResiduals1DOFFitter::kAlign];
    const double residsigma = par[MuonResiduals1DOFFitter::kSigma];
    const double residgamma = par[MuonResiduals1DOFFitter::kGamma];

    double weight = (1. / redchi2) * MuonResiduals1DOFFitter_number_of_hits / MuonResiduals1DOFFitter_sum_of_weights;
    if (!MuonResiduals1DOFFitter_weightAlignment)
      weight = 1.;

    if (!MuonResiduals1DOFFitter_weightAlignment || TMath::Prob(redchi2 * 8, 8) < 0.99) {  // no spikes allowed
      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
        fval += -weight * MuonResidualsFitter_logPureGaussian(residual, residpeak, residsigma);
      } else if (fitter->residualsModel() == MuonResidualsFitter::kPowerLawTails) {
        fval += -weight * MuonResidualsFitter_logPowerLawTails(residual, residpeak, residsigma, residgamma);
      } else if (fitter->residualsModel() == MuonResidualsFitter::kROOTVoigt) {
        fval += -weight * MuonResidualsFitter_logROOTVoigt(residual, residpeak, residsigma, residgamma);
      } else if (fitter->residualsModel() == MuonResidualsFitter::kGaussPowerTails) {
        fval += -weight * MuonResidualsFitter_logGaussPowerTails(residual, residpeak, residsigma);
      } else {
        assert(false);
      }
    }
  }
}

double MuonResiduals1DOFFitter::sumofweights() {
  MuonResiduals1DOFFitter_sum_of_weights = 0.;
  MuonResiduals1DOFFitter_number_of_hits = 0.;
  MuonResiduals1DOFFitter_weightAlignment = m_weightAlignment;
  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    if (m_weightAlignment) {
      double redchi2 = (*resiter)[MuonResiduals1DOFFitter::kRedChi2];
      if (TMath::Prob(redchi2 * 8, 8) < 0.99) {  // no spikes allowed
        MuonResiduals1DOFFitter_sum_of_weights += 1. / redchi2;
        MuonResiduals1DOFFitter_number_of_hits += 1.;
      }
    } else {
      MuonResiduals1DOFFitter_sum_of_weights += 1.;
      MuonResiduals1DOFFitter_number_of_hits += 1.;
    }
  }
  return MuonResiduals1DOFFitter_sum_of_weights;
}

bool MuonResiduals1DOFFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized
  sumofweights();

  double resid_sum = 0.;
  double resid_sum2 = 0.;
  double resid_N = 0.;

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[MuonResiduals1DOFFitter::kResid];
    const double redchi2 = (*resiter)[MuonResiduals1DOFFitter::kRedChi2];
    double weight = 1. / redchi2;
    if (!m_weightAlignment)
      weight = 1.;

    if (!m_weightAlignment || TMath::Prob(redchi2 * 8, 8) < 0.99) {  // no spikes allowed
      if (fabs(residual) < 10.) {                                    // 10 cm
        resid_sum += weight * residual;
        resid_sum2 += weight * residual * residual;
        resid_N += weight;
      }
    }
  }

  double resid_mean = resid_sum / resid_N;
  double resid_stdev = sqrt(resid_sum2 / resid_N - pow(resid_sum / resid_N, 2));

  resid_sum = 0.;
  resid_sum2 = 0.;
  resid_N = 0.;

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double residual = (*resiter)[MuonResiduals1DOFFitter::kResid];
    const double redchi2 = (*resiter)[MuonResiduals1DOFFitter::kRedChi2];
    double weight = 1. / redchi2;
    if (!m_weightAlignment)
      weight = 1.;

    if (!m_weightAlignment || TMath::Prob(redchi2 * 8, 8) < 0.99) {  // no spikes allowed
      if (fabs(residual - resid_mean) < 2.5 * resid_stdev) {
        resid_sum += weight * residual;
        resid_sum2 += weight * residual * residual;
        resid_N += weight;
      }
    }
  }

  resid_mean = resid_sum / resid_N;
  resid_stdev = sqrt(resid_sum2 / resid_N - pow(resid_sum / resid_N, 2));

  std::vector<int> num;
  std::vector<std::string> name;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  if (fixed(kAlign)) {
    num.push_back(kAlign);
    name.push_back(std::string("Align"));
    start.push_back(0.);
    step.push_back(0.01 * resid_stdev);
    low.push_back(0.);
    high.push_back(0.);
  } else {
    num.push_back(kAlign);
    name.push_back(std::string("Align"));
    start.push_back(resid_mean);
    step.push_back(0.01 * resid_stdev);
    low.push_back(0.);
    high.push_back(0.);
  }
  num.push_back(kSigma);
  name.push_back(std::string("Sigma"));
  start.push_back(resid_stdev);
  step.push_back(0.01 * resid_stdev);
  low.push_back(0.);
  high.push_back(0.);
  if (residualsModel() != kPureGaussian && residualsModel() != kGaussPowerTails) {
    num.push_back(kGamma);
    name.push_back(std::string("Gamma"));
    start.push_back(0.1 * resid_stdev);
    step.push_back(0.01 * resid_stdev);
    low.push_back(0.);
    high.push_back(0.);
  }

  return dofit(&MuonResiduals1DOFFitter_FCN, num, name, start, step, low, high);
}

double MuonResiduals1DOFFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  sumofweights();

  std::stringstream name_residual, name_residual_raw;
  name_residual << name << "_residual";
  name_residual_raw << name << "_residual_raw";

  double min_residual = -100.;
  double max_residual = 100.;
  TH1F *hist_residual = dir->make<TH1F>(name_residual.str().c_str(), "", 100, min_residual, max_residual);
  TH1F *hist_residual_raw = dir->make<TH1F>(name_residual_raw.str().c_str(), "", 100, min_residual, max_residual);

  name_residual << "_fit";
  TF1 *fit_residual = nullptr;
  if (residualsModel() == kPureGaussian) {
    fit_residual =
        new TF1(name_residual.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(MuonResiduals1DOFFitter_sum_of_weights * (max_residual - min_residual) / 100.,
                                10. * value(kAlign),
                                10. * value(kSigma));
  } else if (residualsModel() == kPowerLawTails) {
    fit_residual =
        new TF1(name_residual.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(MuonResiduals1DOFFitter_sum_of_weights * (max_residual - min_residual) / 100.,
                                10. * value(kAlign),
                                10. * value(kSigma),
                                10. * value(kGamma));
  } else if (residualsModel() == kROOTVoigt) {
    fit_residual =
        new TF1(name_residual.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(MuonResiduals1DOFFitter_sum_of_weights * (max_residual - min_residual) / 100.,
                                10. * value(kAlign),
                                10. * value(kSigma),
                                10. * value(kGamma));
  } else if (residualsModel() == kGaussPowerTails) {
    fit_residual =
        new TF1(name_residual.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(MuonResiduals1DOFFitter_sum_of_weights * (max_residual - min_residual) / 100.,
                                10. * value(kAlign),
                                10. * value(kSigma));
  } else {
    assert(false);
  }

  fit_residual->SetLineColor(2);
  fit_residual->SetLineWidth(2);
  fit_residual->Write();

  for (std::vector<double *>::const_iterator resiter = residuals_begin(); resiter != residuals_end(); ++resiter) {
    const double resid = (*resiter)[MuonResiduals1DOFFitter::kResid];
    const double redchi2 = (*resiter)[MuonResiduals1DOFFitter::kRedChi2];
    double weight = 1. / redchi2;
    if (!m_weightAlignment)
      weight = 1.;

    if (!m_weightAlignment || TMath::Prob(redchi2 * 8, 8) < 0.99) {  // no spikes allowed
      hist_residual->Fill(10. * (resid + value(kAlign)), weight);
    }

    hist_residual_raw->Fill(10. * resid);
  }

  double chi2 = 0.;
  double ndof = 0.;
  for (int i = 1; i <= hist_residual->GetNbinsX(); i++) {
    double xi = hist_residual->GetBinCenter(i);
    double yi = hist_residual->GetBinContent(i);
    double yerri = hist_residual->GetBinError(i);
    double yth = fit_residual->Eval(xi);
    if (yerri > 0.) {
      chi2 += pow((yth - yi) / yerri, 2);
      ndof += 1.;
    }
  }
  ndof -= npar();

  return (ndof > 0. ? chi2 / ndof : -1.);
}
