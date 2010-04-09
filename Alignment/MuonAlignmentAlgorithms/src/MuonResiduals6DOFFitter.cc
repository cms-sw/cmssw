#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFFitter.h"
#include "TH2F.h"
// #include "TTree.h"
#include "TMath.h"

static TMinuit *MuonResiduals6DOFFitter_TMinuit;
static double MuonResiduals6DOFFitter_sum_of_weights;
static double MuonResiduals6DOFFitter_number_of_hits;
static bool MuonResiduals6DOFFitter_weightAlignment;

void MuonResiduals6DOFFitter::inform(TMinuit *tMinuit) {
  MuonResiduals6DOFFitter_TMinuit = tMinuit;
}

double MuonResiduals6DOFFitter_x(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double alphax, double residual_dxdz) {
  return delta_x - track_dxdz * delta_z - track_y * track_dxdz * delta_phix + track_x * track_dxdz * delta_phiy - track_y * delta_phiz + residual_dxdz * alphax;
}

double MuonResiduals6DOFFitter_y(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz, double alphay, double residual_dydz) {
  return delta_y - track_dydz * delta_z - track_y * track_dydz * delta_phix + track_x * track_dydz * delta_phiy + track_x * delta_phiz + residual_dydz * alphay;
}

double MuonResiduals6DOFFitter_dxdz(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz) {
  return -track_dxdz * track_dydz * delta_phix + (1. + track_dxdz * track_dxdz) * delta_phiy - track_dydz * delta_phiz;
}

double MuonResiduals6DOFFitter_dydz(double delta_x, double delta_y, double delta_z, double delta_phix, double delta_phiy, double delta_phiz, double track_x, double track_y, double track_dxdz, double track_dydz) {
  return -(1. + track_dydz * track_dydz) * delta_phix + track_dxdz * track_dydz * delta_phiy + track_dxdz * delta_phiz;
}

Double_t MuonResiduals6DOFFitter_x_trackx_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_x(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9], par[10], par[11]); }
Double_t MuonResiduals6DOFFitter_x_tracky_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_x(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9], par[10], par[11]); }
Double_t MuonResiduals6DOFFitter_x_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_x(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9], par[10], par[11]); }
Double_t MuonResiduals6DOFFitter_x_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_x(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0], par[10], par[11]); }

Double_t MuonResiduals6DOFFitter_y_trackx_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_y(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9], par[12], par[13]); }
Double_t MuonResiduals6DOFFitter_y_tracky_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_y(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9], par[12], par[13]); }
Double_t MuonResiduals6DOFFitter_y_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_y(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9], par[12], par[13]); }
Double_t MuonResiduals6DOFFitter_y_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 10.*MuonResiduals6DOFFitter_y(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0], par[12], par[13]); }

Double_t MuonResiduals6DOFFitter_dxdz_trackx_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dxdz(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9]); }
Double_t MuonResiduals6DOFFitter_dxdz_tracky_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dxdz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9]); }
Double_t MuonResiduals6DOFFitter_dxdz_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dxdz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9]); }
Double_t MuonResiduals6DOFFitter_dxdz_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dxdz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0]); }

Double_t MuonResiduals6DOFFitter_dydz_trackx_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dydz(par[0], par[1], par[2], par[3], par[4], par[5], xvec[0], par[7], par[8], par[9]); }
Double_t MuonResiduals6DOFFitter_dydz_tracky_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dydz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], xvec[0], par[8], par[9]); }
Double_t MuonResiduals6DOFFitter_dydz_trackdxdz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dydz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], xvec[0], par[9]); }
Double_t MuonResiduals6DOFFitter_dydz_trackdydz_TF1(Double_t *xvec, Double_t *par) { return 1000.*MuonResiduals6DOFFitter_dydz(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], xvec[0]); }

void MuonResiduals6DOFFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) {
  MuonResidualsFitterFitInfo *fitinfo = (MuonResidualsFitterFitInfo*)(MuonResiduals6DOFFitter_TMinuit->GetObjectFit());
  MuonResidualsFitter *fitter = fitinfo->fitter();

  fval = 0.;
  for (std::vector<double*>::const_iterator resiter = fitter->residuals_begin();  resiter != fitter->residuals_end();  ++resiter) {
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

    double residXpeak = MuonResiduals6DOFFitter_x(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, alphax, resslopeX);
    double residYpeak = MuonResiduals6DOFFitter_y(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY, alphay, resslopeY);
    double slopeXpeak = MuonResiduals6DOFFitter_dxdz(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY);
    double slopeYpeak = MuonResiduals6DOFFitter_dydz(alignx, aligny, alignz, alignphix, alignphiy, alignphiz, positionX, positionY, angleX, angleY);

    double weight = (1./redchi2) * MuonResiduals6DOFFitter_number_of_hits / MuonResiduals6DOFFitter_sum_of_weights;
    if (!MuonResiduals6DOFFitter_weightAlignment) weight = 1.;

    if (!MuonResiduals6DOFFitter_weightAlignment  ||  TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed

      if (fitter->residualsModel() == MuonResidualsFitter::kPureGaussian) {
	fval += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
	fval += -weight * MuonResidualsFitter_logPureGaussian(residY, residYpeak, resYsigma);
	fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
	fval += -weight * MuonResidualsFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma);
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
}

double MuonResiduals6DOFFitter::sumofweights() {
  MuonResiduals6DOFFitter_sum_of_weights = 0.;
  MuonResiduals6DOFFitter_number_of_hits = 0.;
  MuonResiduals6DOFFitter_weightAlignment = m_weightAlignment;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    if (m_weightAlignment) {
       double redchi2 = (*resiter)[MuonResiduals6DOFFitter::kRedChi2];
       if (TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed
	  MuonResiduals6DOFFitter_sum_of_weights += 1./redchi2;
	  MuonResiduals6DOFFitter_number_of_hits += 1.;
       }
    }
    else {
       MuonResiduals6DOFFitter_sum_of_weights += 1.;
       MuonResiduals6DOFFitter_number_of_hits += 1.;
    }
  }
  return MuonResiduals6DOFFitter_sum_of_weights;
}

bool MuonResiduals6DOFFitter::fit(Alignable *ali) {
  initialize_table();  // if not already initialized
  sumofweights();

  double residx_mean = 0;
  double residy_mean = 0;
  double resslopex_mean = 0;
  double resslopey_mean = 0;
  double residx_stdev = 0.5;
  double residy_stdev = 3.0;
  double resslopex_stdev = 0.005;
  double resslopey_stdev = 0.03;
  double alphax_estimate = 0;
  double alphay_estimate = 0;

  std::vector<int> num;
  std::vector<std::string> name;
  std::vector<double> start;
  std::vector<double> step;
  std::vector<double> low;
  std::vector<double> high;

  if (fixed(kAlignX)) {
  num.push_back(kAlignX);         name.push_back(std::string("AlignX"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignX);         name.push_back(std::string("AlignX"));         start.push_back(residx_mean);     step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  }
  if (fixed(kAlignY)) {
  num.push_back(kAlignY);         name.push_back(std::string("AlignY"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignY);         name.push_back(std::string("AlignY"));         start.push_back(residy_mean);     step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  }
  num.push_back(kAlignZ);         name.push_back(std::string("AlignZ"));         start.push_back(0.);              step.push_back(0.1);                      low.push_back(0.);   high.push_back(0.);
  if (fixed(kAlignPhiX)) {
  num.push_back(kAlignPhiX);      name.push_back(std::string("AlignPhiX"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignPhiX);      name.push_back(std::string("AlignPhiX"));      start.push_back(-resslopey_mean); step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  }
  if (fixed(kAlignPhiY)) {
  num.push_back(kAlignPhiY);      name.push_back(std::string("AlignPhiY"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  } else {
  num.push_back(kAlignPhiY);      name.push_back(std::string("AlignPhiY"));      start.push_back(resslopex_mean);  step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  }
  num.push_back(kAlignPhiZ);      name.push_back(std::string("AlignPhiZ"));      start.push_back(0.);              step.push_back(0.001);                    low.push_back(0.);   high.push_back(0.);
  num.push_back(kResidXSigma);    name.push_back(std::string("ResidXSigma"));    start.push_back(residx_stdev);    step.push_back(0.01*residx_stdev);        low.push_back(0.);   high.push_back(0.);
  num.push_back(kResidYSigma);    name.push_back(std::string("ResidYSigma"));    start.push_back(residy_stdev);    step.push_back(0.01*residy_stdev);        low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeXSigma); name.push_back(std::string("ResSlopeXSigma")); start.push_back(resslopex_stdev); step.push_back(0.01*resslopex_stdev);     low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeYSigma); name.push_back(std::string("ResSlopeYSigma")); start.push_back(resslopey_stdev); step.push_back(0.01*resslopey_stdev);     low.push_back(0.);   high.push_back(0.);
  num.push_back(kAlphaX);         name.push_back(std::string("AlphaX"));         start.push_back(alphax_estimate); step.push_back(0.01*resslopex_stdev);     low.push_back(0.);   high.push_back(0.);
  num.push_back(kAlphaY);         name.push_back(std::string("AlphaY"));         start.push_back(alphay_estimate); step.push_back(0.01*resslopey_stdev);     low.push_back(0.);   high.push_back(0.);
  if (residualsModel() != kPureGaussian && residualsModel() != kGaussPowerTails) {
  num.push_back(kResidXGamma);    name.push_back(std::string("ResidXGamma"));    start.push_back(0.1*residx_stdev);    step.push_back(0.01*residx_stdev);    low.push_back(0.);   high.push_back(0.);
  num.push_back(kResidYGamma);    name.push_back(std::string("ResidYGamma"));    start.push_back(0.1*residy_stdev);    step.push_back(0.01*residy_stdev);    low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeXGamma); name.push_back(std::string("ResSlopeXGamma")); start.push_back(0.1*resslopex_stdev); step.push_back(0.01*resslopex_stdev); low.push_back(0.);   high.push_back(0.);
  num.push_back(kResSlopeYGamma); name.push_back(std::string("ResSlopeYGamma")); start.push_back(0.1*resslopey_stdev); step.push_back(0.01*resslopey_stdev); low.push_back(0.);   high.push_back(0.);
  }

  return dofit(&MuonResiduals6DOFFitter_FCN, num, name, start, step, low, high);
}

double MuonResiduals6DOFFitter::plot(std::string name, TFileDirectory *dir, Alignable *ali) {
  sumofweights();

//   std::stringstream name_ntuple;
//   name_ntuple << name << "_ntuple";
//   TTree *ntuple = dir->make<TTree>(name_ntuple.str().c_str(), "");
//   Float_t ntuple_residX;
//   Float_t ntuple_residY;
//   Float_t ntuple_resslopeX;
//   Float_t ntuple_resslopeY;
//   Float_t ntuple_positionX;
//   Float_t ntuple_positionY;
//   Float_t ntuple_angleX;
//   Float_t ntuple_angleY;
//   Float_t ntuple_redchi2;
//   Float_t ntuple_prob;

//   ntuple->Branch("residX", &ntuple_residX, "residX/F");
//   ntuple->Branch("residY", &ntuple_residY, "residY/F");
//   ntuple->Branch("resslopeX", &ntuple_resslopeX, "resslopeX/F");
//   ntuple->Branch("resslopeY", &ntuple_resslopeY, "resslopeY/F");
//   ntuple->Branch("positionX", &ntuple_positionX, "positionX/F");
//   ntuple->Branch("positionY", &ntuple_positionY, "positionY/F");
//   ntuple->Branch("angleX", &ntuple_angleX, "angleX/F");
//   ntuple->Branch("angleY", &ntuple_angleY, "angleY/F");
//   ntuple->Branch("redchi2", &ntuple_redchi2, "redchi2/F");
//   ntuple->Branch("prob", &ntuple_prob, "prob/F");
  
//   for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
//     ntuple_residX = (*resiter)[MuonResiduals6DOFFitter::kResidX];
//     ntuple_residY = (*resiter)[MuonResiduals6DOFFitter::kResidY];
//     ntuple_resslopeX = (*resiter)[MuonResiduals6DOFFitter::kResSlopeX];
//     ntuple_resslopeY = (*resiter)[MuonResiduals6DOFFitter::kResSlopeY];
//     ntuple_positionX = (*resiter)[MuonResiduals6DOFFitter::kPositionX];
//     ntuple_positionY = (*resiter)[MuonResiduals6DOFFitter::kPositionY];
//     ntuple_angleX = (*resiter)[MuonResiduals6DOFFitter::kAngleX];
//     ntuple_angleY = (*resiter)[MuonResiduals6DOFFitter::kAngleY];
//     ntuple_redchi2 = (*resiter)[MuonResiduals6DOFFitter::kRedChi2];
//     ntuple_prob = TMath::Prob((*resiter)[MuonResiduals6DOFFitter::kRedChi2]);

//     ntuple->Fill();
//   }

  std::stringstream name_x, name_y, name_dxdz, name_dydz, name_x_raw, name_y_raw, name_dxdz_raw, name_dydz_raw, name_x_cut, name_y_cut, name_alphax, name_alphay;
  std::stringstream name_x_trackx, name_y_trackx, name_dxdz_trackx, name_dydz_trackx;
  std::stringstream name_x_tracky, name_y_tracky, name_dxdz_tracky, name_dydz_tracky;
  std::stringstream name_x_trackdxdz, name_y_trackdxdz, name_dxdz_trackdxdz, name_dydz_trackdxdz;
  std::stringstream name_x_trackdydz, name_y_trackdydz, name_dxdz_trackdydz, name_dydz_trackdydz;

  name_x << name << "_x";
  name_y << name << "_y";
  name_dxdz << name << "_dxdz";
  name_dydz << name << "_dydz";
  name_x_raw << name << "_x_raw";
  name_y_raw << name << "_y_raw";
  name_dxdz_raw << name << "_dxdz_raw";
  name_dydz_raw << name << "_dydz_raw";
  name_x_cut << name << "_x_cut";
  name_y_cut << name << "_y_cut";
  name_alphax << name << "_alphax";
  name_alphay << name << "_alphay";
  name_x_trackx << name << "_x_trackx";
  name_y_trackx << name << "_y_trackx";
  name_dxdz_trackx << name << "_dxdz_trackx";
  name_dydz_trackx << name << "_dydz_trackx";
  name_x_tracky << name << "_x_tracky";
  name_y_tracky << name << "_y_tracky";
  name_dxdz_tracky << name << "_dxdz_tracky";
  name_dydz_tracky << name << "_dydz_tracky";
  name_x_trackdxdz << name << "_x_trackdxdz";
  name_y_trackdxdz << name << "_y_trackdxdz";
  name_dxdz_trackdxdz << name << "_dxdz_trackdxdz";
  name_dydz_trackdxdz << name << "_dydz_trackdxdz";
  name_x_trackdydz << name << "_x_trackdydz";
  name_y_trackdydz << name << "_y_trackdydz";
  name_dxdz_trackdydz << name << "_dxdz_trackdydz";
  name_dydz_trackdydz << name << "_dydz_trackdydz";

  double width = ali->surface().width();
  double length = ali->surface().length();
  double min_x = -100.;            double max_x = 100.;
  double min_y = -200.;            double max_y = 200.;
  double min_dxdz = -100.;         double max_dxdz = 100.;
  double min_dydz = -200.;         double max_dydz = 200.;
  double min_trackx = -width/2.;   double max_trackx = width/2.;
  double min_tracky = -length/2.;  double max_tracky = length/2.;
  double min_trackdxdz = -1.5;     double max_trackdxdz = 1.5;
  double min_trackdydz = -1.5;     double max_trackdydz = 1.5;

  TH1F *hist_x = dir->make<TH1F>(name_x.str().c_str(), "", 100, min_x, max_x);
  TH1F *hist_y = dir->make<TH1F>(name_y.str().c_str(), "", 100, min_y, max_y);
  TH1F *hist_dxdz = dir->make<TH1F>(name_dxdz.str().c_str(), "", 100, min_dxdz, max_dxdz);
  TH1F *hist_dydz = dir->make<TH1F>(name_dydz.str().c_str(), "", 100, min_dydz, max_dydz);
  TH1F *hist_x_raw = dir->make<TH1F>(name_x_raw.str().c_str(), "", 100, min_x, max_x);
  TH1F *hist_y_raw = dir->make<TH1F>(name_y_raw.str().c_str(), "", 100, min_y, max_y);
  TH1F *hist_dxdz_raw = dir->make<TH1F>(name_dxdz_raw.str().c_str(), "", 100, min_dxdz, max_dxdz);
  TH1F *hist_dydz_raw = dir->make<TH1F>(name_dydz_raw.str().c_str(), "", 100, min_dydz, max_dydz);
  TH1F *hist_x_cut = dir->make<TH1F>(name_x_cut.str().c_str(), "", 100, min_x, max_x);
  TH1F *hist_y_cut = dir->make<TH1F>(name_y_cut.str().c_str(), "", 100, min_y, max_y);
  TH2F *hist_alphax = dir->make<TH2F>(name_alphax.str().c_str(), "", 40, min_dxdz, max_dxdz, 40, -20., 20.);
  TH2F *hist_alphay = dir->make<TH2F>(name_alphay.str().c_str(), "", 40, min_dydz, max_dydz, 40, -100., 100.);
  TProfile *hist_x_trackx = dir->make<TProfile>(name_x_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_x, max_x);
  TProfile *hist_y_trackx = dir->make<TProfile>(name_y_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_y, max_y);
  TProfile *hist_dxdz_trackx = dir->make<TProfile>(name_dxdz_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackx = dir->make<TProfile>(name_dydz_trackx.str().c_str(), "", 100, min_trackx, max_trackx, min_dydz, max_dydz);
  TProfile *hist_x_tracky = dir->make<TProfile>(name_x_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_x, max_x);
  TProfile *hist_y_tracky = dir->make<TProfile>(name_y_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_y, max_y);
  TProfile *hist_dxdz_tracky = dir->make<TProfile>(name_dxdz_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_dxdz, max_dxdz);
  TProfile *hist_dydz_tracky = dir->make<TProfile>(name_dydz_tracky.str().c_str(), "", 100, min_tracky, max_tracky, min_dydz, max_dydz);
  TProfile *hist_x_trackdxdz = dir->make<TProfile>(name_x_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_x, max_x);
  TProfile *hist_y_trackdxdz = dir->make<TProfile>(name_y_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_y, max_y);
  TProfile *hist_dxdz_trackdxdz = dir->make<TProfile>(name_dxdz_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackdxdz = dir->make<TProfile>(name_dydz_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz, min_dydz, max_dydz);
  TProfile *hist_x_trackdydz = dir->make<TProfile>(name_x_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_x, max_x);
  TProfile *hist_y_trackdydz = dir->make<TProfile>(name_y_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_y, max_y);
  TProfile *hist_dxdz_trackdydz = dir->make<TProfile>(name_dxdz_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_dxdz, max_dxdz);
  TProfile *hist_dydz_trackdydz = dir->make<TProfile>(name_dydz_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz, min_dydz, max_dydz);

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

  name_x << "_fit";
  name_y << "_fit";
  name_dxdz << "_fit";
  name_dydz << "_fit";
  name_alphax << "_fit";
  name_alphay << "_fit";
  name_x_trackx << "_fit";
  name_y_trackx << "_fit";
  name_dxdz_trackx << "_fit";
  name_dydz_trackx << "_fit";
  name_x_tracky << "_fit";
  name_y_tracky << "_fit";
  name_dxdz_tracky << "_fit";
  name_dydz_tracky << "_fit";
  name_x_trackdxdz << "_fit";
  name_y_trackdxdz << "_fit";
  name_dxdz_trackdxdz << "_fit";
  name_dydz_trackdxdz << "_fit";
  name_x_trackdydz << "_fit";
  name_y_trackdydz << "_fit";
  name_dxdz_trackdydz << "_fit";
  name_dydz_trackdydz << "_fit";

  TF1 *fit_x = NULL;
  TF1 *fit_y = NULL;
  TF1 *fit_dxdz = NULL;
  TF1 *fit_dydz = NULL;
  if (residualsModel() == kPureGaussian) {
    fit_x = new TF1(name_x.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_x, max_x, 3);
    fit_x->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_x - min_x)/100., 10.*value(kAlignX), 10.*value(kResidXSigma));
    fit_y = new TF1(name_y.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_y, max_y, 3);
    fit_y->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_y - min_y)/100., 10.*value(kAlignY), 10.*value(kResidYSigma));
    fit_dxdz = new TF1(name_dxdz.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_dxdz, max_dxdz, 3);
    fit_dxdz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dxdz - min_dxdz)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma));
    fit_dydz = new TF1(name_dydz.str().c_str(), MuonResidualsFitter_pureGaussian_TF1, min_dydz, max_dydz, 3);
    fit_dydz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dydz - min_dydz)/100., -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma));
  }
  else if (residualsModel() == kPowerLawTails) {
    fit_x = new TF1(name_x.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_x, max_x, 4);
    fit_x->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_x - min_x)/100., 10.*value(kAlignX), 10.*value(kResidXSigma), 10.*value(kResidXGamma));
    fit_y = new TF1(name_y.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_y, max_y, 4);
    fit_y->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_y - min_y)/100., 10.*value(kAlignY), 10.*value(kResidYSigma), 10.*value(kResidYGamma));
    fit_dxdz = new TF1(name_dxdz.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_dxdz, max_dxdz, 4);
    fit_dxdz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dxdz - min_dxdz)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma), 1000.*value(kResSlopeXGamma));
    fit_dydz = new TF1(name_dydz.str().c_str(), MuonResidualsFitter_powerLawTails_TF1, min_dydz, max_dydz, 4);
    fit_dydz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dydz - min_dydz)/100., -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma), 1000.*value(kResSlopeYGamma));
  }
  else if (residualsModel() == kROOTVoigt) {
    fit_x = new TF1(name_x.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_x, max_x, 4);
    fit_x->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_x - min_x)/100., 10.*value(kAlignX), 10.*value(kResidXSigma), 10.*value(kResidXGamma));
    fit_y = new TF1(name_y.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_y, max_y, 4);
    fit_y->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_y - min_y)/100., 10.*value(kAlignY), 10.*value(kResidYSigma), 10.*value(kResidYGamma));
    fit_dxdz = new TF1(name_dxdz.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_dxdz, max_dxdz, 4);
    fit_dxdz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dxdz - min_dxdz)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma), 1000.*value(kResSlopeXGamma));
    fit_dydz = new TF1(name_dydz.str().c_str(), MuonResidualsFitter_ROOTVoigt_TF1, min_dydz, max_dydz, 4);
    fit_dydz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dydz - min_dydz)/100., -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma), 1000.*value(kResSlopeYGamma));
  }
  else if (residualsModel() == kGaussPowerTails) {
    fit_x = new TF1(name_x.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_x, max_x, 3);
    fit_x->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_x - min_x)/100., 10.*value(kAlignX), 10.*value(kResidXSigma));
    fit_y = new TF1(name_y.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_y, max_y, 3);
    fit_y->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_y - min_y)/100., 10.*value(kAlignY), 10.*value(kResidYSigma));
    fit_dxdz = new TF1(name_dxdz.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_dxdz, max_dxdz, 3);
    fit_dxdz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dxdz - min_dxdz)/100., 1000.*value(kAlignPhiY), 1000.*value(kResSlopeXSigma));
    fit_dydz = new TF1(name_dydz.str().c_str(), MuonResidualsFitter_GaussPowerTails_TF1, min_dydz, max_dydz, 3);
    fit_dydz->SetParameters(MuonResiduals6DOFFitter_sum_of_weights * (max_dydz - min_dydz)/100., -1000.*value(kAlignPhiX), 1000.*value(kResSlopeYSigma));
  }
  else { assert(false); }

  fit_x->SetLineColor(2);     fit_x->SetLineWidth(2);
  fit_y->SetLineColor(2);     fit_y->SetLineWidth(2);
  fit_dxdz->SetLineColor(2);  fit_dxdz->SetLineWidth(2);
  fit_dydz->SetLineColor(2);  fit_dydz->SetLineWidth(2);
  fit_x->Write();
  fit_y->Write();
  fit_dxdz->Write();
  fit_dydz->Write();

  TF1 *fit_alphax = new TF1(name_alphax.str().c_str(), "[0] + x*[1]", min_dxdz, max_dxdz);
  fit_alphax->SetParameters(10.*value(kAlignX), 10.*value(kAlphaX)/1000.);
  TF1 *fit_alphay = new TF1(name_alphay.str().c_str(), "[0] + x*[1]", min_dydz, max_dydz);
  fit_alphay->SetParameters(10.*value(kAlignY), 10.*value(kAlphaY)/1000.);

  fit_alphax->SetLineColor(2);  fit_alphax->SetLineWidth(2);
  fit_alphay->SetLineColor(2);  fit_alphay->SetLineWidth(2);
  fit_alphax->Write();
  fit_alphay->Write();

  TProfile *fit_x_trackx = dir->make<TProfile>(name_x_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_y_trackx = dir->make<TProfile>(name_y_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_dxdz_trackx = dir->make<TProfile>(name_dxdz_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_dydz_trackx = dir->make<TProfile>(name_dydz_trackx.str().c_str(), "", 100, min_trackx, max_trackx);
  TProfile *fit_x_tracky = dir->make<TProfile>(name_x_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_y_tracky = dir->make<TProfile>(name_y_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_dxdz_tracky = dir->make<TProfile>(name_dxdz_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_dydz_tracky = dir->make<TProfile>(name_dydz_tracky.str().c_str(), "", 100, min_tracky, max_tracky);
  TProfile *fit_x_trackdxdz = dir->make<TProfile>(name_x_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_y_trackdxdz = dir->make<TProfile>(name_y_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_dxdz_trackdxdz = dir->make<TProfile>(name_dxdz_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_dydz_trackdxdz = dir->make<TProfile>(name_dydz_trackdxdz.str().c_str(), "", 500, min_trackdxdz, max_trackdxdz);
  TProfile *fit_x_trackdydz = dir->make<TProfile>(name_x_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_y_trackdydz = dir->make<TProfile>(name_y_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_dxdz_trackdydz = dir->make<TProfile>(name_dxdz_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);
  TProfile *fit_dydz_trackdydz = dir->make<TProfile>(name_dydz_trackdydz.str().c_str(), "", 500, min_trackdydz, max_trackdydz);

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

  name_x_trackx << "line";
  name_y_trackx << "line";
  name_dxdz_trackx << "line";
  name_dydz_trackx << "line";
  name_x_tracky << "line";
  name_y_tracky << "line";
  name_dxdz_tracky << "line";
  name_dydz_tracky << "line";
  name_x_trackdxdz << "line";
  name_y_trackdxdz << "line";
  name_dxdz_trackdxdz << "line";
  name_dydz_trackdxdz << "line";
  name_x_trackdydz << "line";
  name_y_trackdydz << "line";
  name_dxdz_trackdydz << "line";
  name_dydz_trackdydz << "line";

  TF1 *fitline_x_trackx = new TF1(name_x_trackx.str().c_str(), MuonResiduals6DOFFitter_x_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_y_trackx = new TF1(name_y_trackx.str().c_str(), MuonResiduals6DOFFitter_y_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_dxdz_trackx = new TF1(name_dxdz_trackx.str().c_str(), MuonResiduals6DOFFitter_dxdz_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_dydz_trackx = new TF1(name_dydz_trackx.str().c_str(), MuonResiduals6DOFFitter_dydz_trackx_TF1, min_trackx, max_trackx, 14);
  TF1 *fitline_x_tracky = new TF1(name_x_tracky.str().c_str(), MuonResiduals6DOFFitter_x_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_y_tracky = new TF1(name_y_tracky.str().c_str(), MuonResiduals6DOFFitter_y_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_dxdz_tracky = new TF1(name_dxdz_tracky.str().c_str(), MuonResiduals6DOFFitter_dxdz_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_dydz_tracky = new TF1(name_dydz_tracky.str().c_str(), MuonResiduals6DOFFitter_dydz_tracky_TF1, min_tracky, max_tracky, 14);
  TF1 *fitline_x_trackdxdz = new TF1(name_x_trackdxdz.str().c_str(), MuonResiduals6DOFFitter_x_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_y_trackdxdz = new TF1(name_y_trackdxdz.str().c_str(), MuonResiduals6DOFFitter_y_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_dxdz_trackdxdz = new TF1(name_dxdz_trackdxdz.str().c_str(), MuonResiduals6DOFFitter_dxdz_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_dydz_trackdxdz = new TF1(name_dydz_trackdxdz.str().c_str(), MuonResiduals6DOFFitter_dydz_trackdxdz_TF1, min_trackdxdz, max_trackdxdz, 14);
  TF1 *fitline_x_trackdydz = new TF1(name_x_trackdydz.str().c_str(), MuonResiduals6DOFFitter_x_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_y_trackdydz = new TF1(name_y_trackdydz.str().c_str(), MuonResiduals6DOFFitter_y_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_dxdz_trackdydz = new TF1(name_dxdz_trackdydz.str().c_str(), MuonResiduals6DOFFitter_dxdz_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);
  TF1 *fitline_dydz_trackdydz = new TF1(name_dydz_trackdydz.str().c_str(), MuonResiduals6DOFFitter_dydz_trackdydz_TF1, min_trackdydz, max_trackdydz, 14);

  double sum_resslopex = 0.;
  double sum_resslopey = 0.;
  double sum_trackx = 0.;
  double sum_tracky = 0.;
  double sum_trackdxdz = 0.;
  double sum_trackdydz = 0.;
  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double resslopeX = (*resiter)[MuonResiduals6DOFFitter::kResSlopeX];
    const double resslopeY = (*resiter)[MuonResiduals6DOFFitter::kResSlopeY];
    const double positionX = (*resiter)[MuonResiduals6DOFFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFFitter::kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed
      sum_resslopex += weight * resslopeX;
      sum_resslopey += weight * resslopeY;
      sum_trackx += weight * positionX;
      sum_tracky += weight * positionY;
      sum_trackdxdz += weight * angleX;
      sum_trackdydz += weight * angleY;
    }
  }
  double mean_resslopex = sum_resslopex / MuonResiduals6DOFFitter_sum_of_weights;
  double mean_resslopey = sum_resslopey / MuonResiduals6DOFFitter_sum_of_weights;
  double mean_trackx = sum_trackx / MuonResiduals6DOFFitter_sum_of_weights;
  double mean_tracky = sum_tracky / MuonResiduals6DOFFitter_sum_of_weights;
  double mean_trackdxdz = sum_trackdxdz / MuonResiduals6DOFFitter_sum_of_weights;
  double mean_trackdydz = sum_trackdydz / MuonResiduals6DOFFitter_sum_of_weights;

  double fitparameters[14];
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
  fitparameters[10] = value(kAlphaX);
  fitparameters[11] = mean_resslopex;
  fitparameters[12] = value(kAlphaY);
  fitparameters[13] = mean_resslopey;

  fitline_x_trackx->SetParameters(fitparameters);
  fitline_y_trackx->SetParameters(fitparameters);
  fitline_dxdz_trackx->SetParameters(fitparameters);
  fitline_dydz_trackx->SetParameters(fitparameters);
  fitline_x_tracky->SetParameters(fitparameters);
  fitline_y_tracky->SetParameters(fitparameters);
  fitline_dxdz_tracky->SetParameters(fitparameters);
  fitline_dydz_tracky->SetParameters(fitparameters);
  fitline_x_trackdxdz->SetParameters(fitparameters);
  fitline_y_trackdxdz->SetParameters(fitparameters);
  fitline_dxdz_trackdxdz->SetParameters(fitparameters);
  fitline_dydz_trackdxdz->SetParameters(fitparameters);
  fitline_x_trackdydz->SetParameters(fitparameters);
  fitline_y_trackdydz->SetParameters(fitparameters);
  fitline_dxdz_trackdydz->SetParameters(fitparameters);
  fitline_dydz_trackdydz->SetParameters(fitparameters);

  fitline_x_trackx->SetLineColor(2);        fitline_x_trackx->SetLineWidth(2);
  fitline_y_trackx->SetLineColor(2);        fitline_y_trackx->SetLineWidth(2);
  fitline_dxdz_trackx->SetLineColor(2);     fitline_dxdz_trackx->SetLineWidth(2);
  fitline_dydz_trackx->SetLineColor(2);     fitline_dydz_trackx->SetLineWidth(2);
  fitline_x_tracky->SetLineColor(2);        fitline_x_tracky->SetLineWidth(2);
  fitline_y_tracky->SetLineColor(2);        fitline_y_tracky->SetLineWidth(2);
  fitline_dxdz_tracky->SetLineColor(2);     fitline_dxdz_tracky->SetLineWidth(2);
  fitline_dydz_tracky->SetLineColor(2);     fitline_dydz_tracky->SetLineWidth(2);
  fitline_x_trackdxdz->SetLineColor(2);     fitline_x_trackdxdz->SetLineWidth(2);
  fitline_y_trackdxdz->SetLineColor(2);     fitline_y_trackdxdz->SetLineWidth(2);
  fitline_dxdz_trackdxdz->SetLineColor(2);  fitline_dxdz_trackdxdz->SetLineWidth(2);
  fitline_dydz_trackdxdz->SetLineColor(2);  fitline_dydz_trackdxdz->SetLineWidth(2);
  fitline_x_trackdydz->SetLineColor(2);     fitline_x_trackdydz->SetLineWidth(2);
  fitline_y_trackdydz->SetLineColor(2);     fitline_y_trackdydz->SetLineWidth(2);
  fitline_dxdz_trackdydz->SetLineColor(2);  fitline_dxdz_trackdydz->SetLineWidth(2);
  fitline_dydz_trackdydz->SetLineColor(2);  fitline_dydz_trackdydz->SetLineWidth(2);

  fitline_x_trackx->Write();
  fitline_y_trackx->Write();
  fitline_dxdz_trackx->Write();
  fitline_dydz_trackx->Write();
  fitline_x_tracky->Write();
  fitline_y_tracky->Write();
  fitline_dxdz_tracky->Write();
  fitline_dydz_tracky->Write();
  fitline_x_trackdxdz->Write();
  fitline_y_trackdxdz->Write();
  fitline_dxdz_trackdxdz->Write();
  fitline_dydz_trackdxdz->Write();
  fitline_x_trackdydz->Write();
  fitline_y_trackdydz->Write();
  fitline_dxdz_trackdydz->Write();
  fitline_dydz_trackdydz->Write();

  for (std::vector<double*>::const_iterator resiter = residuals_begin();  resiter != residuals_end();  ++resiter) {
    const double residX = (*resiter)[MuonResiduals6DOFFitter::kResidX];
    const double residY = (*resiter)[MuonResiduals6DOFFitter::kResidY];
    const double resslopeX = (*resiter)[MuonResiduals6DOFFitter::kResSlopeX];
    const double resslopeY = (*resiter)[MuonResiduals6DOFFitter::kResSlopeY];
    const double positionX = (*resiter)[MuonResiduals6DOFFitter::kPositionX];
    const double positionY = (*resiter)[MuonResiduals6DOFFitter::kPositionY];
    const double angleX = (*resiter)[MuonResiduals6DOFFitter::kAngleX];
    const double angleY = (*resiter)[MuonResiduals6DOFFitter::kAngleY];
    const double redchi2 = (*resiter)[MuonResiduals6DOFFitter::kRedChi2];
    double weight = 1./redchi2;
    if (!m_weightAlignment) weight = 1.;

    if (!m_weightAlignment  ||  TMath::Prob(redchi2*12, 12) < 0.99) {  // no spikes allowed

      hist_alphax->Fill(1000.*resslopeX, 10.*residX);
      hist_alphay->Fill(1000.*resslopeY, 10.*residY);

      double geom_residX = MuonResiduals6DOFFitter_x(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, value(kAlphaX), resslopeX);
      hist_x->Fill(10.*(residX - geom_residX + value(kAlignX)), weight);
      hist_x_trackx->Fill(positionX, 10.*residX, weight);
      hist_x_tracky->Fill(positionY, 10.*residX, weight);
      hist_x_trackdxdz->Fill(angleX, 10.*residX, weight);
      hist_x_trackdydz->Fill(angleY, 10.*residX, weight);
      fit_x_trackx->Fill(positionX, 10.*geom_residX, weight);
      fit_x_tracky->Fill(positionY, 10.*geom_residX, weight);
      fit_x_trackdxdz->Fill(angleX, 10.*geom_residX, weight);
      fit_x_trackdydz->Fill(angleY, 10.*geom_residX, weight);

      double geom_residY = MuonResiduals6DOFFitter_y(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY, value(kAlphaY), resslopeY);
      hist_y->Fill(10.*(residY - geom_residY + value(kAlignY)), weight);
      hist_y_trackx->Fill(positionX, 10.*residY, weight);
      hist_y_tracky->Fill(positionY, 10.*residY, weight);
      hist_y_trackdxdz->Fill(angleX, 10.*residY, weight);
      hist_y_trackdydz->Fill(angleY, 10.*residY, weight);
      fit_y_trackx->Fill(positionX, 10.*geom_residY, weight);
      fit_y_tracky->Fill(positionY, 10.*geom_residY, weight);
      fit_y_trackdxdz->Fill(angleX, 10.*geom_residY, weight);
      fit_y_trackdydz->Fill(angleY, 10.*geom_residY, weight);

      double geom_resslopeX = MuonResiduals6DOFFitter_dxdz(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY);
      hist_dxdz->Fill(1000.*(resslopeX - geom_resslopeX + value(kAlignPhiY)), weight);
      hist_dxdz_trackx->Fill(positionX, 1000.*resslopeX, weight);
      hist_dxdz_tracky->Fill(positionY, 1000.*resslopeX, weight);
      hist_dxdz_trackdxdz->Fill(angleX, 1000.*resslopeX, weight);
      hist_dxdz_trackdydz->Fill(angleY, 1000.*resslopeX, weight);
      fit_dxdz_trackx->Fill(positionX, 1000.*geom_resslopeX, weight);
      fit_dxdz_tracky->Fill(positionY, 1000.*geom_resslopeX, weight);
      fit_dxdz_trackdxdz->Fill(angleX, 1000.*geom_resslopeX, weight);
      fit_dxdz_trackdydz->Fill(angleY, 1000.*geom_resslopeX, weight);

      double geom_resslopeY = MuonResiduals6DOFFitter_dydz(value(kAlignX), value(kAlignY), value(kAlignZ), value(kAlignPhiX), value(kAlignPhiY), value(kAlignPhiZ), positionX, positionY, angleX, angleY);
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
