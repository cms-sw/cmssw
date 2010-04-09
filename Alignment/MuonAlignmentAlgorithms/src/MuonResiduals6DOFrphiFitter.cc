#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFrphiFitter.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TH2F.h"
#include "TMath.h"

static TMinuit *MuonResiduals6DOFrphiFitter_TMinuit;
static double MuonResiduals6DOFrphiFitter_sum_of_weights;
static double MuonResiduals6DOFrphiFitter_number_of_hits;
static bool MuonResiduals6DOFrphiFitter_weightAlignment;

const CSCGeometry *MuonResiduals6DOFrphiFitter_cscGeometry;
static CSCDetId MuonResiduals6DOFrphiFitter_cscDetId;
static double MuonResiduals6DOFrphiFitter_R;

void MuonResiduals6DOFrphiFitter::inform(TMinuit *tMinuit) {
  MuonResiduals6DOFrphiFitter_TMinuit = tMinuit;
}

double MuonResiduals6DOFrphiFitter_residual(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double R, double alpha, double resslope) {
  return delta_x - (track_x/R - 3.*pow(track_x/R, 3)) * delta_y - track_dxdz * delta_z - track_y * track_dxdz * delta_phix + track_x * track_dxdz * delta_phiy - track_y * delta_phiz + resslope * alpha;
}

double MuonResiduals6DOFrphiFitter_resslope(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double R) {
  return -track_dxdz/2./R * delta_y + (track_x/R - track_dxdz * track_dydz) * delta_phix + (1. + track_dxdz * track_dxdz) * delta_phiy - track_dydz * delta_phiz;
}

double MuonResiduals6DOFrphiFitter_effectiveR(double x, double y) {
//   CSCDetId id = MuonResiduals6DOFrphiFitter_cscDetId;
//   CSCDetId layerId(id.endcap(), id.station(), id.ring(), id.chamber(), 3);

//   int strip = MuonResiduals6DOFrphiFitter_cscGeometry->layer(layerId)->geometry()->nearestStrip(LocalPoint(x, y, 0.));
//   double angle = MuonResiduals6DOFrphiFitter_cscGeometry->layer(layerId)->geometry()->stripAngle(strip) - M_PI/2.;

//   if (fabs(angle) < 1e-10) return MuonResiduals6DOFrphiFitter_R;
//   else return x/tan(angle) - y;

  return MuonResiduals6DOFrphiFitter_R;
}

Double_t MuonResiduals6DOFrphiFitter_residual_trackx_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFrphiFitter_residual(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9], MuonResiduals6DOFrphiFitter_effectiveR(xvec[0], par[7]), par[10], par[11]); }
Double_t MuonResiduals6DOFrphiFitter_residual_tracky_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFrphiFitter_residual(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9], MuonResiduals6DOFrphiFitter_effectiveR(par[6], xvec[0]), par[10], par[11]); }
Double_t MuonResiduals6DOFrphiFitter_residual_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFrphiFitter_residual(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9], MuonResiduals6DOFrphiFitter_effectiveR(par[6], par[7]), par[10], par[11]); }
Double_t MuonResiduals6DOFrphiFitter_residual_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFrphiFitter_residual(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0], MuonResiduals6DOFrphiFitter_effectiveR(par[6], par[7]), par[10], par[11]); }

Double_t MuonResiduals6DOFrphiFitter_resslope_trackx_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFrphiFitter_resslope(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9], MuonResiduals6DOFrphiFitter_effectiveR(xvec[0], par[7])); }
Double_t MuonResiduals6DOFrphiFitter_resslope_tracky_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFrphiFitter_resslope(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9], MuonResiduals6DOFrphiFitter_effectiveR(par[6], xvec[0])); }
Double_t MuonResiduals6DOFrphiFitter_resslope_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFrphiFitter_resslope(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9], MuonResiduals6DOFrphiFitter_effectiveR(par[6], par[7])); }
Double_t MuonResiduals6DOFrphiFitter_resslope_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFrphiFitter_resslope(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0], MuonResiduals6DOFrphiFitter_effectiveR(par[6], par[7])); }

void MuonResiduals6DOFrphiFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(MuonResiduals6DOFrphiFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter) {
    const double residual = (*resiter)[MuonResiduals6DOFrphiFitter::kResid];
    const double resslope = (*resiter)[MuonResiduals6DOFrphiFitter::kResSlope];
    const double positionX = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFrphiFitter::kRedChi2];

    const double alignx = par[MuonResiduals6DOFrphiFitter::kAlignX];
    const double aligny = par[MuonResiduals6DOFrphiFitter::kAlignY];
    const double alignz = par[MuonResiduals6DOFrphiFitter::kAlignZ];
    const double alignphix = par[MuonResiduals6DOFrphiFitter::kAlignPhiX];
    const double alignphiy = par[MuonResiduals6DOFrphiFitter::kAlignPhiY];
    const double alignphiz = par[MuonResiduals6DOFrphiFitter::kAlignPhiZ];
    const double residsigma = par[MuonResiduals6DOFrphiFitter::kResidSigma];
    const double resslopesigma = par[MuonResiduals6DOFrphiFitter::kResSlopeSigma];
    const double alpha = par[MuonResiduals6DOFrphiFitter::kAlpha];
    const double residgamma = par[MuonResiduals6DOFrphiFitter::kResidGamma];
    const double resslopegamma = par[MuonResiduals6DOFrphiFitter::kResSlopeGamma];

    double residpeak = MuonResiduals6DOFrphiFitter_residual(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, MuonResiduals6DOFrphiFitter_effectiveR(positionX, positionY), alpha, resslope);
    double resslopepeak = MuonResiduals6DOFrphiFitter_resslope(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, MuonResiduals6DOFrphiFitter_effectiveR(positionX, positionY));

    double weight = (1./redchi2) * MuonResiduals6DOFrphiFitter_number_of_hits / MuonResiduals6DOFrphiFitter_sum_of_weights;
    if (!MuonResiduals6DOFrphiFitter_weightAlignment) weight = 1.;

    if (!MuonResiduals6DOFrphiFitter_weightAlignment  ||  TMath::Prob(redchi2*6, 6) < 0.99) {  // no spikes allowed

      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
	fval += -weight * MuonResidualsFitter_logPureGaussian(residual, residpeak, residsigma);
	fval += -weight * MuonResidualsFitter_logPureGaussian(resslope, resslopepeak, resslopesigma);
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

double MuonResiduals6DOFrphiFitter::sumofweights() {
  MuonResiduals6DOFrphiFitter_sum_of_weights = 0.;
  MuonResiduals6DOFrphiFitter_number_of_hits = 0.;
  MuonResiduals6DOFrphiFitter_weightAlignment = m_weightAlignment;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    if (m_weightAlignment) {
       double redchi2 = (*resiter)[MuonResiduals6DOFrphiFitter::kRedChi2];
       if (TMath::Prob(redchi2*6, 6) < 0.99) {  // no spikes allowed
	  MuonResiduals6DOFrphiFitter_sum_of_weights += 1./redchi2;
	  MuonResiduals6DOFrphiFitter_number_of_hits += 1.;
       }
    }
    else {
       MuonResiduals6DOFrphiFitter_sum_of_weights += 1.;
       MuonResiduals6DOFrphiFitter_number_of_hits += 1.;
    }
  }
  return MuonResiduals6DOFrphiFitter_sum_of_weights;
}

bool MuonResiduals6DOFrphiFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized
  sumofweights();

  double resid_mean = 0;
  double resslope_mean = 0;
  double resid_stdev = 0.5;
  double resslope_stdev = 0.005;
  double alpha_estimate = 0;

  std::vector<int> num;
  std::vector<std::string> name;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  if (fixed(kAlignX)) {
  num.push_back(kAlignX);         name.push_back(std::string("AlignX"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignX);         name.push_back(std::string("AlignX"));         start.push_back(resid_mean);      step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  }
  num.push_back(kAlignY);         name.push_back(std::string("AlignY"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  num.push_back(kAlignZ);         name.push_back(std::string("AlignZ"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  num.push_back(kAlignPhiX);      name.push_back(std::string("AlignPhiX"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  if (fixed(kAlignPhiY)) {
  num.push_back(kAlignPhiY);      name.push_back(std::string("AlignPhiY"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignPhiY);      name.push_back(std::string("AlignPhiY"));      start.push_back(resslope_mean);   step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  }
  num.push_back(kAlignPhiZ);      name.push_back(std::string("AlignPhiZ"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  num.push_back(kResidSigma);     name.push_back(std::string("ResidSigma"));     start.push_back(resid_stdev);     step.push_back(0.01*resid_stdev);         low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeSigma);  name.push_back(std::string("ResSlopeSigma"));  start.push_back(resslope_stdev);  step.push_back(0.01*resslope_stdev);      low.push_back(0.);   high.push_back(0.);
  num.push_back(kAlpha);          name.push_back(std::string("Alpha"));          start.push_back(alpha_estimate);  step.push_back(0.01*resslope_stdev);      low.push_back(0.);   high.push_back(0.);
  if (residualsModel() != kPureGaussian && residualsModel() != kGaussPowerTails) {
  num.push_back(kResidGamma);     name.push_back(std::string("ResidGamma"));     start.push_back(0.1*resid_stdev);     step.push_back(0.01*resid_stdev);     low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeGamma);  name.push_back(std::string("ResSlopeGamma"));  start.push_back(0.1*resslope_stdev);  step.push_back(0.01*resslope_stdev);  low.push_back(0.);   high.push_back(0.);
  }

  return dofit(&MuonResiduals6DOFrphiFitter_FCN, num, name, start, step, low, high);
}

double MuonResiduals6DOFrphiFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  sumofweights();

  MuonResiduals6DOFrphiFitter_R = sqrt(pow(ali->globalPosition().x(), 2) + pow(ali->globalPosition().y(), 2));
  MuonResiduals6DOFrphiFitter_cscGeometry = m_cscGeometry;
  MuonResiduals6DOFrphiFitter_cscDetId = CSCDetId(ali->geomDetId().rawId());

  std::stringstream name_residual, name_resslope, name_residual_raw, name_resslope_raw, name_residual_cut, name_alpha;
  std::stringstream name_residual_trackx, name_resslope_trackx;
  std::stringstream name_residual_tracky, name_resslope_tracky;
  std::stringstream name_residual_trackdxdz, name_resslope_trackdxdz;
  std::stringstream name_residual_trackdydz, name_resslope_trackdydz;

  name_residual << name << "_residual";
  name_resslope << name << "_resslope";
  name_residual_raw << name << "_residual_raw";
  name_resslope_raw << name << "_resslope_raw";
  name_residual_cut << name << "_residual_cut";
  name_alpha << name << "_alpha";
  name_residual_trackx << name << "_residual_trackx";
  name_resslope_trackx << name << "_resslope_trackx";
  name_residual_tracky << name << "_residual_tracky";
  name_resslope_tracky << name << "_resslope_tracky";
  name_residual_trackdxdz << name << "_residual_trackdxdz";
  name_resslope_trackdxdz << name << "_resslope_trackdxdz";
  name_residual_trackdydz << name << "_residual_trackdydz";
  name_resslope_trackdydz << name << "_resslope_trackdydz";

  double width = ali->surface().width();
  double length = ali->surface().length();
  double min_residual = -100.;     double max_residual = 100.;
  double min_resslope = -100.;     double max_resslope = 100.;
  double min_trackx = -width/2.;   double max_trackx = width/2.;
  double min_tracky = -length/2.;  double max_tracky = length/2.;
  double min_trackdxdz = -1.5;     double max_trackdxdz = 1.5;
  double min_trackdydz = -1.5;     double max_trackdydz = 1.5;

  TH1F *hist_residual = dir->make<TH1F>(name_residual.str().c_str(), "", 100, min_residual, max_residual);
  TH1F *hist_resslope = dir->make<TH1F>(name_resslope.str().c_str(), "", 100, min_resslope, max_resslope);
  TH1F *hist_residual_raw = dir->make<TH1F>(name_residual_raw.str().c_str(), "", 100, min_residual, max_residual);
  TH1F *hist_resslope_raw = dir->make<TH1F>(name_resslope_raw.str().c_str(), "", 100, min_resslope, max_resslope);
  TH1F *hist_residual_cut = dir->make<TH1F>(name_residual_cut.str().c_str(), "", 100, min_residual, max_residual);
  TH2F *hist_alpha = dir->make<TH2F>(name_alpha.str().c_str(),"", 40, min_resslope, max_resslope, 40, -20., 20.);
  TProfile *hist_residual_trackx = dir->make<TProfile>(name_residual_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_residual, max_residual);
  TProfile *hist_resslope_trackx = dir->make<TProfile>(name_resslope_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_resslope, max_resslope);
  TProfile *hist_residual_tracky = dir->make<TProfile>(name_residual_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_residual, max_residual);
  TProfile *hist_resslope_tracky = dir->make<TProfile>(name_resslope_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_resslope, max_resslope);
  TProfile *hist_residual_trackdxdz = dir->make<TProfile>(name_residual_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_residual, max_residual);
  TProfile *hist_resslope_trackdxdz = dir->make<TProfile>(name_resslope_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_resslope, max_resslope);
  TProfile *hist_residual_trackdydz = dir->make<TProfile>(name_residual_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_residual, max_residual);
  TProfile *hist_resslope_trackdydz = dir->make<TProfile>(name_resslope_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_resslope, max_resslope);

  hist_residual_trackx->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackx->SetAxisRange(-10., 10., "Y");
  hist_residual_tracky->SetAxisRange(-10., 10., "Y");
  hist_resslope_tracky->SetAxisRange(-10., 10., "Y");
  hist_residual_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackdxdz->SetAxisRange(-10., 10., "Y");
  hist_residual_trackdydz->SetAxisRange(-10., 10., "Y");
  hist_resslope_trackdydz->SetAxisRange(-10., 10., "Y");

  name_residual << "_fit";
  name_resslope << "_fit";
  name_alpha << "_fit";
  name_residual_trackx << "_fit";
  name_resslope_trackx << "_fit";
  name_residual_tracky << "_fit";
  name_resslope_tracky << "_fit";
  name_residual_trackdxdz << "_fit";
  name_resslope_trackdxdz << "_fit";
  name_residual_trackdydz << "_fit";
  name_resslope_trackdydz << "_fit";

  TF1 *fit_residual = NULL;
  TF1 *fit_resslope = NULL;
  if (residualsModel() == kPureGaussian) {
    fit_residual = new TF1(name_residual.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_residual - min_residual)/100., 10.*value(kAlignX), 10.*value(kResidSigma));
    fit_resslope = new TF1(name_resslope.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_resslope, max_resslope, 3);
    fit_resslope->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_resslope - min_resslope)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma));
  }
  else if (residualsModel() == kPowerLawTails) {
    fit_residual = new TF1(name_residual.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_residual - min_residual)/100., 10.*value(kAlignX), 10.*value(kResidSigma), 10.*value(kResidGamma));
    fit_resslope = new TF1(name_resslope.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_resslope, max_resslope, 4);
    fit_resslope->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_resslope - min_resslope)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma), 1000.*value(kResSlopeGamma));
  }
  else if (residualsModel() == kROOTVoigt) {
    fit_residual = new TF1(name_residual.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_residual, max_residual, 4);
    fit_residual->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_residual - min_residual)/100., 10.*value(kAlignX), 10.*value(kResidSigma), 10.*value(kResidGamma));
    fit_resslope = new TF1(name_resslope.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_resslope, max_resslope, 4);
    fit_resslope->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_resslope - min_resslope)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma), 1000.*value(kResSlopeGamma));
  }
  else if (residualsModel() == kGaussPowerTails) {
    fit_residual = new TF1(name_residual.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_residual, max_residual, 3);
    fit_residual->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_residual - min_residual)/100., 10.*value(kAlignX), 10.*value(kResidSigma));
    fit_resslope = new TF1(name_resslope.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_resslope, max_resslope, 3);
    fit_resslope->SetParameters(MuonResiduals6DOFrphiFitter_sum_of_weights * (max_resslope - min_resslope)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeSigma));
  }
  else { assert(false); }

  fit_residual->SetLineColor(2);  fit_residual->SetLineWidth(2);
  fit_resslope->SetLineColor(2);  fit_resslope->SetLineWidth(2);
  fit_residual->Write();
  fit_resslope->Write();

  TF1 *fit_alpha = new TF1(name_alpha.str().c_str(), "[0] + x*[1]", min_resslope, max_resslope);
  fit_alpha->SetParameters(10.*value(kAlignX), 10.*value(kAlpha)/1000.);
  fit_alpha->SetLineColor(2);  fit_alpha->SetLineWidth(2);
  fit_alpha->Write();

  TProfile *fit_residual_trackx = dir->make<TProfile>(name_residual_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_resslope_trackx = dir->make<TProfile>(name_resslope_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_residual_tracky = dir->make<TProfile>(name_residual_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_resslope_tracky = dir->make<TProfile>(name_resslope_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_residual_trackdxdz = dir->make<TProfile>(name_residual_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_resslope_trackdxdz = dir->make<TProfile>(name_resslope_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_residual_trackdydz = dir->make<TProfile>(name_residual_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_resslope_trackdydz = dir->make<TProfile>(name_resslope_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);

  fit_residual_trackx->SetLineColor(2);     fit_residual_trackx->SetLineWidth(2);
  fit_resslope_trackx->SetLineColor(2);     fit_resslope_trackx->SetLineWidth(2);
  fit_residual_tracky->SetLineColor(2);     fit_residual_tracky->SetLineWidth(2);
  fit_resslope_tracky->SetLineColor(2);     fit_resslope_tracky->SetLineWidth(2);
  fit_residual_trackdxdz->SetLineColor(2);  fit_residual_trackdxdz->SetLineWidth(2);
  fit_resslope_trackdxdz->SetLineColor(2);  fit_resslope_trackdxdz->SetLineWidth(2);
  fit_residual_trackdydz->SetLineColor(2);  fit_residual_trackdydz->SetLineWidth(2);
  fit_resslope_trackdydz->SetLineColor(2);  fit_resslope_trackdydz->SetLineWidth(2);

  name_residual_trackx << "line";
  name_resslope_trackx << "line";
  name_residual_tracky << "line";
  name_resslope_tracky << "line";
  name_residual_trackdxdz << "line";
  name_resslope_trackdxdz << "line";
  name_residual_trackdydz << "line";
  name_resslope_trackdydz << "line";

  TF1 *fitline_residual_trackx = new TF1(name_residual_trackx.str().c_str(), MuonResiduals6DOFrphiFitter_residual_trackx_TF1, min_trackx, max_trackx, 12);
  TF1 *fitline_resslope_trackx = new TF1(name_resslope_trackx.str().c_str(), MuonResiduals6DOFrphiFitter_resslope_trackx_TF1, min_trackx, max_trackx, 12);
  TF1 *fitline_residual_tracky = new TF1(name_residual_tracky.str().c_str(), MuonResiduals6DOFrphiFitter_residual_tracky_TF1, min_tracky, max_tracky, 12);
  TF1 *fitline_resslope_tracky = new TF1(name_resslope_tracky.str().c_str(), MuonResiduals6DOFrphiFitter_resslope_tracky_TF1, min_tracky, max_tracky, 12);
  TF1 *fitline_residual_trackdxdz = new TF1(name_residual_trackdxdz.str().c_str(), MuonResiduals6DOFrphiFitter_residual_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 12);
  TF1 *fitline_resslope_trackdxdz = new TF1(name_resslope_trackdxdz.str().c_str(), MuonResiduals6DOFrphiFitter_resslope_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 12);
  TF1 *fitline_residual_trackdydz = new TF1(name_residual_trackdydz.str().c_str(), MuonResiduals6DOFrphiFitter_residual_trackdydz_TF1, min_trackdydz, max_trackdydz, 12);
  TF1 *fitline_resslope_trackdydz = new TF1(name_resslope_trackdydz.str().c_str(), MuonResiduals6DOFrphiFitter_resslope_trackdydz_TF1, min_trackdydz, max_trackdydz, 12);

  double sum_resslope = 0.;
  double sum_trackx = 0.;
  double sum_tracky = 0.;
  double sum_trackdxdz = 0.;
  double sum_trackdydz = 0.;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double resslope = (*resiter)[MuonResiduals6DOFrphiFitter::kResSlope];
    const double positionX = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFrphiFitter::kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*6, 6) < 0.99) {  // no spikes allowed
      sum_resslope += weight * resslope;
      sum_trackx += weight * positionX;
      sum_tracky += weight * positionY;
      sum_trackdxdz += weight * angleX;
      sum_trackdydz += weight * angleY;
    }
  }
  double mean_resslope = sum_resslope / MuonResiduals6DOFrphiFitter_sum_of_weights;
  double mean_trackx = sum_trackx / MuonResiduals6DOFrphiFitter_sum_of_weights;
  double mean_tracky = sum_tracky / MuonResiduals6DOFrphiFitter_sum_of_weights;
  double mean_trackdxdz = sum_trackdxdz / MuonResiduals6DOFrphiFitter_sum_of_weights;
  double mean_trackdydz = sum_trackdydz / MuonResiduals6DOFrphiFitter_sum_of_weights;

  double fitparameters[12];
  fitparameters[0] = value(kAlignX);
  fitparameters[1] = value(kAlignY);
  fitparameters[2] = value(kAlignZ);
  fitparameters[3] = value(kAlignPhiX);
  fitparameters[4] = value(kAlignPhiY);
  fitparameters[5] = value(kAlignPhiZ);
  fitparameters[6] = mean_trackx;
  fitparameters[7] = mean_tracky;
  fitparameters[8] = mean_trackdxdz;
  fitparameters[9] = mean_trackdydz;
  fitparameters[10] = value(kAlpha);
  fitparameters[11] = mean_resslope;

  fitline_residual_trackx->SetParameters(fitparameters);
  fitline_resslope_trackx->SetParameters(fitparameters);
  fitline_residual_tracky->SetParameters(fitparameters);
  fitline_resslope_tracky->SetParameters(fitparameters);
  fitline_residual_trackdxdz->SetParameters(fitparameters);
  fitline_resslope_trackdxdz->SetParameters(fitparameters);
  fitline_residual_trackdydz->SetParameters(fitparameters);
  fitline_resslope_trackdydz->SetParameters(fitparameters);

  fitline_residual_trackx->SetLineColor(2);        fitline_residual_trackx->SetLineWidth(2);
  fitline_resslope_trackx->SetLineColor(2);        fitline_resslope_trackx->SetLineWidth(2);
  fitline_residual_tracky->SetLineColor(2);        fitline_residual_tracky->SetLineWidth(2);
  fitline_resslope_tracky->SetLineColor(2);        fitline_resslope_tracky->SetLineWidth(2);
  fitline_residual_trackdxdz->SetLineColor(2);     fitline_residual_trackdxdz->SetLineWidth(2);
  fitline_resslope_trackdxdz->SetLineColor(2);     fitline_resslope_trackdxdz->SetLineWidth(2);
  fitline_residual_trackdydz->SetLineColor(2);     fitline_residual_trackdydz->SetLineWidth(2);
  fitline_resslope_trackdydz->SetLineColor(2);     fitline_resslope_trackdydz->SetLineWidth(2);

  fitline_residual_trackx->Write();
  fitline_resslope_trackx->Write();
  fitline_residual_tracky->Write();
  fitline_resslope_tracky->Write();
  fitline_residual_trackdxdz->Write();
  fitline_resslope_trackdxdz->Write();
  fitline_residual_trackdydz->Write();
  fitline_resslope_trackdydz->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double resid = (*resiter)[MuonResiduals6DOFrphiFitter::kResid];
    const double resslope = (*resiter)[MuonResiduals6DOFrphiFitter::kResSlope];
    const double positionX = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFrphiFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFrphiFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFrphiFitter::kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*6, 6) < 0.99) {  // no spikes allowed
      hist_alpha->Fill(1000.*resslope, 10.*resid);

      double geom_resid = MuonResiduals6DOFrphiFitter_residual(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, MuonResiduals6DOFrphiFitter_effectiveR(positionX, positionY), value(kAlpha), resslope);
      hist_residual->Fill(10.*(resid - geom_resid + value(kAlignX)), weight);
      hist_residual_trackx->Fill(positionX, 10.*resid, weight);
      hist_residual_tracky->Fill(positionY, 10.*resid, weight);
      hist_residual_trackdxdz->Fill(angleX, 10.*resid, weight);
      hist_residual_trackdydz->Fill(angleY, 10.*resid, weight);
      fit_residual_trackx->Fill(positionX, 10.*geom_resid, weight);
      fit_residual_tracky->Fill(positionY, 10.*geom_resid, weight);
      fit_residual_trackdxdz->Fill(angleX, 10.*geom_resid, weight);
      fit_residual_trackdydz->Fill(angleY, 10.*geom_resid, weight);

      double geom_resslope = MuonResiduals6DOFrphiFitter_resslope(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, MuonResiduals6DOFrphiFitter_effectiveR(positionX, positionY));
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
