/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include <dirent.h>
#include <map>
#include <string>

#include "TFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

TDirectory *mkdir(TDirectory *parent, const char *child) {
  TDirectory *dir = (TDirectory *)parent->Get(child);
  if (!dir)
    dir = parent->mkdir(child);
  return dir;
}

//----------------------------------------------------------------------------------------------------

struct Stat {
  double s1, sx, sxx, sxxx, sxxxx;

  Stat() : s1(0.), sx(0.), sxx(0.), sxxx(0.), sxxxx(0.) {}

  void fill(double x) {
    s1 += 1;
    sx += x;
    sxx += x * x;
    sxxx += x * x * x;
    sxxxx += x * x * x * x;
  }

  double n() const { return s1; }

  double mean() const { return sx / s1; }

  double meanError() const {
    double v = rms() / sqrt(s1);
    return v;
  }

  double rms() const {
    double sig_sq = (sxx - sx * sx / s1) / (s1 - 1.);
    if (sig_sq < 0)
      sig_sq = 0;

    return sqrt(sig_sq);
  }

  double rmsError() const {
    // see R.J.Barlow: Statistics, page 78
    double mu = mean();
    double E2 = rms();
    E2 *= E2;
    //if (print) printf("\t\t%E, %E, %E, %E, %E\n", sxxxx, -4.*sxxx*mu, 6.*sxx*mu*mu, -4.*sx*mu*mu*mu, s1*mu*mu*mu*mu);
    double E4 = (sxxxx - 4. * sxxx * mu + 6. * sxx * mu * mu - 4. * sx * mu * mu * mu + s1 * mu * mu * mu * mu) / s1;
    //if (print) printf("\t\tmu = %E, E2 = %E, E4 = %E\n", mu, E2, E4);
    double v_sig_sq = (s1 - 1) * ((s1 - 1) * E4 - (s1 - 3) * E2 * E2) / s1 / s1 / s1;
    double v_sig = v_sig_sq / 4. / E2;
    //if (print) printf("\t\tv_sig_sq = %E\n", v_sig_sq);
    if (v_sig < 0 || std::isnan(v_sig) || std::isinf(v_sig)) {
      //printf(">> Stat::rmsError > ERROR: v_siq = %E < 0.\n", v_sig);
      v_sig = 0;
    }
    double v = sqrt(v_sig);
    return v;
  }
};

//----------------------------------------------------------------------------------------------------

struct DetStat {
  // _v ... value given by Jan algorithm
  // _u ... uncertainty estimated by Jan algorithm
  // _e ... difference (Jan - Ideal) algorithm
  Stat sh_x_v, sh_x_u, sh_x_e;
  Stat sh_y_v, sh_y_u, sh_y_e;
  Stat rot_z_v, rot_z_u, rot_z_e;
  Stat sh_z_v, sh_z_u, sh_z_e;
};

//----------------------------------------------------------------------------------------------------

struct RPStat {
  Stat sh_x_v, sh_x_u, sh_x_e;
  Stat sh_y_v, sh_y_u, sh_y_e;
  Stat rot_z_v, rot_z_u, rot_z_e;
  Stat sh_z_v, sh_z_u, sh_z_e;
};

//----------------------------------------------------------------------------------------------------

struct Desc {
  unsigned int N;  // number of events
  unsigned int i;  // iteration
  unsigned int d;  // detector
  Desc(unsigned int _N, unsigned int _i, unsigned int _d) : N(_N), i(_i), d(_d) {}
  bool operator<(const Desc &c) const;
};

bool Desc::operator<(const Desc &c) const {
  if (this->N < c.N)
    return true;
  if (this->N > c.N)
    return false;
  if (this->i < c.i)
    return true;
  if (this->i > c.i)
    return false;
  if (this->d < c.d)
    return true;
  if (this->d > c.d)
    return false;
  return false;
}

//----------------------------------------------------------------------------------------------------

map<Desc, DetStat> det_stat;
map<Desc, RPStat> rp_stat;

//----------------------------------------------------------------------------------------------------

void resetStatistics() {
  det_stat.clear();
  rp_stat.clear();
}

//----------------------------------------------------------------------------------------------------

void updateSensorStatistics(unsigned int n_events,
                            unsigned iteration,
                            const CTPPSRPAlignmentCorrectionsData &r_actual,
                            const CTPPSRPAlignmentCorrectionsData &r_ideal) {
  for (CTPPSRPAlignmentCorrectionsData::mapType::const_iterator it = r_actual.getSensorMap().begin();
       it != r_actual.getSensorMap().end();
       ++it) {
    unsigned int id = it->first;
    const auto &c_actual = r_actual.getFullSensorCorrection(id, false);
    const auto &c_ideal = r_ideal.getFullSensorCorrection(id, false);

    DetStat &s = det_stat[Desc(n_events, iteration, id)];

    s.sh_x_v.fill(c_actual.getShX());
    s.sh_x_u.fill(c_actual.getShXUnc());
    s.sh_x_e.fill(c_actual.getShX() - c_ideal.getShX());

    s.sh_y_v.fill(c_actual.getShY());
    s.sh_y_u.fill(c_actual.getShYUnc());
    s.sh_y_e.fill(c_actual.getShY() - c_ideal.getShY());

    s.sh_z_v.fill(c_actual.getShZ());
    s.sh_z_u.fill(c_actual.getShZUnc());
    s.sh_z_e.fill(c_actual.getShZ() - c_ideal.getShZ());

    s.rot_z_v.fill(c_actual.getRotZ());
    s.rot_z_u.fill(c_actual.getRotZUnc());
    s.rot_z_e.fill(c_actual.getRotZ() - c_ideal.getRotZ());
  }
}

//----------------------------------------------------------------------------------------------------

void updateRPStatistics(unsigned int n_events,
                        unsigned iteration,
                        const CTPPSRPAlignmentCorrectionsData &r_actual,
                        const CTPPSRPAlignmentCorrectionsData &r_ideal) {
  for (CTPPSRPAlignmentCorrectionsData::mapType::const_iterator it = r_actual.getRPMap().begin();
       it != r_actual.getRPMap().end();
       ++it) {
    unsigned int id = it->first;
    const auto &c_actual = r_actual.getRPCorrection(id);
    const auto &c_ideal = r_ideal.getRPCorrection(id);

    RPStat &s = rp_stat[Desc(n_events, iteration, id)];

    s.sh_x_v.fill(c_actual.getShX());
    s.sh_x_u.fill(c_actual.getShXUnc());
    s.sh_x_e.fill(c_actual.getShX() - c_ideal.getShX());

    s.sh_y_v.fill(c_actual.getShY());
    s.sh_y_u.fill(c_actual.getShYUnc());
    s.sh_y_e.fill(c_actual.getShY() - c_ideal.getShY());

    s.sh_z_v.fill(c_actual.getShZ());
    s.sh_z_u.fill(c_actual.getShZUnc());
    s.sh_z_e.fill(c_actual.getShZ() - c_ideal.getShZ());

    s.rot_z_v.fill(c_actual.getRotZ());
    s.rot_z_u.fill(c_actual.getRotZUnc());
    s.rot_z_e.fill(c_actual.getRotZ() - c_ideal.getRotZ());
  }
}

//----------------------------------------------------------------------------------------------------

struct StatGraphs {
  TGraph *n;
  TGraphErrors *v_m, *v_v, *u_m, *u_v, *e_m, *e_v, *eR;
  StatGraphs()
      : n(new TGraph()),
        v_m(new TGraphErrors()),
        v_v(new TGraphErrors()),
        u_m(new TGraphErrors()),
        u_v(new TGraphErrors()),
        e_m(new TGraphErrors()),
        e_v(new TGraphErrors()),
        eR(new TGraphErrors()) {}

  void write(const char *xLabel);
};

//----------------------------------------------------------------------------------------------------

#define ENTRY(tag, label)             \
  tag->SetName(#tag);                 \
  sprintf(buf, ";%s;" label, xLabel); \
  tag->SetTitle(buf);                 \
  tag->Write();

void StatGraphs::write(const char *xLabel) {
  char buf[50];
  ENTRY(n, "number of repetitions");
  ENTRY(v_m, "value mean");
  ENTRY(v_v, "value variation");
  ENTRY(u_m, "estim. uncertainty mean");
  ENTRY(u_v, "estim. uncertainty variation");
  ENTRY(e_m, "error mean");
  ENTRY(e_v, "error variation");
  ENTRY(eR, "error variation / estim. uncertainty");
}

//----------------------------------------------------------------------------------------------------

struct DetGraphs {
  StatGraphs sh_x, sh_y, rot_z, sh_z;
  void fill(double x, const DetStat &);
  void write(const char *xLabel);
};

//----------------------------------------------------------------------------------------------------

double eR_error(const Stat &e, const Stat &u) {
  double a = e.rms(), ae = e.rmsError();
  double b = u.mean(), be = u.meanError();

  return (b <= 0) ? 0. : a / b * sqrt(ae * ae / a / a + be * be / b / b);
}

//----------------------------------------------------------------------------------------------------

void DetGraphs::fill(double x, const DetStat &s) {
  int idx = sh_x.n->GetN();

  sh_x.n->SetPoint(idx, x, s.sh_x_u.n());
  sh_x.v_m->SetPoint(idx, x, s.sh_x_v.mean());
  sh_x.v_v->SetPoint(idx, x, s.sh_x_v.rms());
  sh_x.u_m->SetPoint(idx, x, s.sh_x_u.mean());
  sh_x.u_v->SetPoint(idx, x, s.sh_x_u.rms());
  sh_x.e_m->SetPoint(idx, x, s.sh_x_e.mean());
  sh_x.e_v->SetPoint(idx, x, s.sh_x_e.rms());
  sh_x.eR->SetPoint(idx, x, s.sh_x_e.rms() / s.sh_x_u.mean());

  sh_x.v_m->SetPointError(idx, 0., s.sh_x_v.meanError());
  sh_x.v_v->SetPointError(idx, 0., s.sh_x_v.rmsError());
  sh_x.u_m->SetPointError(idx, 0., s.sh_x_u.meanError());
  sh_x.u_v->SetPointError(idx, 0., s.sh_x_u.rmsError());
  sh_x.e_m->SetPointError(idx, 0., s.sh_x_e.meanError());
  sh_x.e_v->SetPointError(idx, 0., s.sh_x_e.rmsError());
  sh_x.eR->SetPointError(idx, 0., eR_error(s.sh_x_e, s.sh_x_u));

  sh_y.n->SetPoint(idx, x, s.sh_y_u.n());
  sh_y.v_m->SetPoint(idx, x, s.sh_y_v.mean());
  sh_y.v_v->SetPoint(idx, x, s.sh_y_v.rms());
  sh_y.u_m->SetPoint(idx, x, s.sh_y_u.mean());
  sh_y.u_v->SetPoint(idx, x, s.sh_y_u.rms());
  sh_y.e_m->SetPoint(idx, x, s.sh_y_e.mean());
  sh_y.e_v->SetPoint(idx, x, s.sh_y_e.rms());
  sh_y.eR->SetPoint(idx, x, s.sh_y_e.rms() / s.sh_y_u.mean());

  sh_y.v_m->SetPointError(idx, 0., s.sh_y_v.meanError());
  sh_y.v_v->SetPointError(idx, 0., s.sh_y_v.rmsError());
  sh_y.u_m->SetPointError(idx, 0., s.sh_y_u.meanError());
  sh_y.u_v->SetPointError(idx, 0., s.sh_y_u.rmsError());
  sh_y.e_m->SetPointError(idx, 0., s.sh_y_e.meanError());
  sh_y.e_v->SetPointError(idx, 0., s.sh_y_e.rmsError());
  sh_y.eR->SetPointError(idx, 0., eR_error(s.sh_y_e, s.sh_y_u));

  rot_z.n->SetPoint(idx, x, s.rot_z_u.n());
  rot_z.v_m->SetPoint(idx, x, s.rot_z_v.mean());
  rot_z.v_v->SetPoint(idx, x, s.rot_z_v.rms());
  rot_z.u_m->SetPoint(idx, x, s.rot_z_u.mean());
  rot_z.u_v->SetPoint(idx, x, s.rot_z_u.rms());
  rot_z.e_m->SetPoint(idx, x, s.rot_z_e.mean());
  rot_z.e_v->SetPoint(idx, x, s.rot_z_e.rms());
  rot_z.eR->SetPoint(idx, x, s.rot_z_e.rms() / s.rot_z_u.mean());

  rot_z.v_m->SetPointError(idx, 0., s.rot_z_v.meanError());
  rot_z.v_v->SetPointError(idx, 0., s.rot_z_v.rmsError());
  rot_z.u_m->SetPointError(idx, 0., s.rot_z_u.meanError());
  rot_z.u_v->SetPointError(idx, 0., s.rot_z_u.rmsError());
  rot_z.e_m->SetPointError(idx, 0., s.rot_z_e.meanError());
  rot_z.e_v->SetPointError(idx, 0., s.rot_z_e.rmsError());
  rot_z.eR->SetPointError(idx, 0., eR_error(s.rot_z_e, s.rot_z_u));

  sh_z.n->SetPoint(idx, x, s.sh_z_u.n());
  sh_z.v_m->SetPoint(idx, x, s.sh_z_v.mean());
  sh_z.v_v->SetPoint(idx, x, s.sh_z_v.rms());
  sh_z.u_m->SetPoint(idx, x, s.sh_z_u.mean());
  sh_z.u_v->SetPoint(idx, x, s.sh_z_u.rms());
  sh_z.e_m->SetPoint(idx, x, s.sh_z_e.mean());
  sh_z.e_v->SetPoint(idx, x, s.sh_z_e.rms());
  sh_z.eR->SetPoint(idx, x, s.sh_z_e.rms() / s.sh_z_u.mean());

  sh_z.v_m->SetPointError(idx, 0., s.sh_z_v.meanError());
  sh_z.v_v->SetPointError(idx, 0., s.sh_z_v.rmsError());
  sh_z.u_m->SetPointError(idx, 0., s.sh_z_u.meanError());
  sh_z.u_v->SetPointError(idx, 0., s.sh_z_u.rmsError());
  sh_z.e_m->SetPointError(idx, 0., s.sh_z_e.meanError());
  sh_z.e_v->SetPointError(idx, 0., s.sh_z_e.rmsError());
  sh_z.eR->SetPointError(idx, 0., eR_error(s.sh_z_e, s.sh_z_u));
}

//----------------------------------------------------------------------------------------------------

void DetGraphs::write(const char *xLabel) {
  gDirectory = mkdir(gDirectory, "sh_x");
  sh_x.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "sh_y");
  sh_y.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "rot_z");
  rot_z.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "sh_z");
  sh_z.write(xLabel);
  gDirectory->cd("..");
}
//----------------------------------------------------------------------------------------------------

struct RPGraphs {
  StatGraphs sh_x, sh_y, rot_z, sh_z;
  void fill(double x, const RPStat &);
  void write(const char *xLabel);
};

//----------------------------------------------------------------------------------------------------

void RPGraphs::fill(double x, const RPStat &s) {
  int idx = sh_x.n->GetN();

  sh_x.n->SetPoint(idx, x, s.sh_x_u.n());
  sh_x.v_m->SetPoint(idx, x, s.sh_x_v.mean());
  sh_x.v_v->SetPoint(idx, x, s.sh_x_v.rms());
  sh_x.u_m->SetPoint(idx, x, s.sh_x_u.mean());
  sh_x.u_v->SetPoint(idx, x, s.sh_x_u.rms());
  sh_x.e_m->SetPoint(idx, x, s.sh_x_e.mean());
  sh_x.e_v->SetPoint(idx, x, s.sh_x_e.rms());
  sh_x.eR->SetPoint(idx, x, s.sh_x_e.rms() / s.sh_x_u.mean());

  sh_x.v_m->SetPointError(idx, 0., s.sh_x_v.meanError());
  sh_x.v_v->SetPointError(idx, 0., s.sh_x_v.rmsError());
  sh_x.u_m->SetPointError(idx, 0., s.sh_x_u.meanError());
  sh_x.u_v->SetPointError(idx, 0., s.sh_x_u.rmsError());
  sh_x.e_m->SetPointError(idx, 0., s.sh_x_e.meanError());
  sh_x.e_v->SetPointError(idx, 0., s.sh_x_e.rmsError());
  sh_x.eR->SetPointError(idx, 0., eR_error(s.sh_x_e, s.sh_x_u));

  sh_y.n->SetPoint(idx, x, s.sh_y_u.n());
  sh_y.v_m->SetPoint(idx, x, s.sh_y_v.mean());
  sh_y.v_v->SetPoint(idx, x, s.sh_y_v.rms());
  sh_y.u_m->SetPoint(idx, x, s.sh_y_u.mean());
  sh_y.u_v->SetPoint(idx, x, s.sh_y_u.rms());
  sh_y.e_m->SetPoint(idx, x, s.sh_y_e.mean());
  sh_y.e_v->SetPoint(idx, x, s.sh_y_e.rms());
  sh_y.eR->SetPoint(idx, x, s.sh_y_e.rms() / s.sh_y_u.mean());

  sh_y.v_m->SetPointError(idx, 0., s.sh_y_v.meanError());
  sh_y.v_v->SetPointError(idx, 0., s.sh_y_v.rmsError());
  sh_y.u_m->SetPointError(idx, 0., s.sh_y_u.meanError());
  sh_y.u_v->SetPointError(idx, 0., s.sh_y_u.rmsError());
  sh_y.e_m->SetPointError(idx, 0., s.sh_y_e.meanError());
  sh_y.e_v->SetPointError(idx, 0., s.sh_y_e.rmsError());
  sh_y.eR->SetPointError(idx, 0., eR_error(s.sh_y_e, s.sh_y_u));

  rot_z.n->SetPoint(idx, x, s.rot_z_u.n());
  rot_z.v_m->SetPoint(idx, x, s.rot_z_v.mean());
  rot_z.v_v->SetPoint(idx, x, s.rot_z_v.rms());
  rot_z.u_m->SetPoint(idx, x, s.rot_z_u.mean());
  rot_z.u_v->SetPoint(idx, x, s.rot_z_u.rms());
  rot_z.e_m->SetPoint(idx, x, s.rot_z_e.mean());
  rot_z.e_v->SetPoint(idx, x, s.rot_z_e.rms());
  rot_z.eR->SetPoint(idx, x, s.rot_z_e.rms() / s.rot_z_u.mean());

  rot_z.v_m->SetPointError(idx, 0., s.rot_z_v.meanError());
  rot_z.v_v->SetPointError(idx, 0., s.rot_z_v.rmsError());
  rot_z.u_m->SetPointError(idx, 0., s.rot_z_u.meanError());
  rot_z.u_v->SetPointError(idx, 0., s.rot_z_u.rmsError());
  rot_z.e_m->SetPointError(idx, 0., s.rot_z_e.meanError());
  rot_z.e_v->SetPointError(idx, 0., s.rot_z_e.rmsError());
  rot_z.eR->SetPointError(idx, 0., eR_error(s.rot_z_e, s.rot_z_u));

  sh_z.n->SetPoint(idx, x, s.sh_z_u.n());
  sh_z.v_m->SetPoint(idx, x, s.sh_z_v.mean());
  sh_z.v_v->SetPoint(idx, x, s.sh_z_v.rms());
  sh_z.u_m->SetPoint(idx, x, s.sh_z_u.mean());
  sh_z.u_v->SetPoint(idx, x, s.sh_z_u.rms());
  sh_z.e_m->SetPoint(idx, x, s.sh_z_e.mean());
  sh_z.e_v->SetPoint(idx, x, s.sh_z_e.rms());
  sh_z.eR->SetPoint(idx, x, s.sh_z_e.rms() / s.sh_z_u.mean());

  sh_z.v_m->SetPointError(idx, 0., s.sh_z_v.meanError());
  sh_z.v_v->SetPointError(idx, 0., s.sh_z_v.rmsError());
  sh_z.u_m->SetPointError(idx, 0., s.sh_z_u.meanError());
  sh_z.u_v->SetPointError(idx, 0., s.sh_z_u.rmsError());
  sh_z.e_m->SetPointError(idx, 0., s.sh_z_e.meanError());
  sh_z.e_v->SetPointError(idx, 0., s.sh_z_e.rmsError());
  sh_z.eR->SetPointError(idx, 0., eR_error(s.sh_z_e, s.sh_z_u));
}

//----------------------------------------------------------------------------------------------------

void RPGraphs::write(const char *xLabel) {
  gDirectory = mkdir(gDirectory, "sh_x");
  sh_x.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "sh_y");
  sh_y.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "rot_z");
  rot_z.write(xLabel);
  gDirectory->cd("..");
  gDirectory = mkdir(gDirectory, "sh_z");
  sh_z.write(xLabel);
  gDirectory->cd("..");
}

//----------------------------------------------------------------------------------------------------

TFile *sf;  // output file

void writeGraphs(string r, string o, string d) {
  map<Desc, DetGraphs> dFcnN;   // here Desc::N is 0 by definition
  map<Desc, DetGraphs> dFcnIt;  // here Desc::i is 0 by definition
  for (map<Desc, DetStat>::iterator it = det_stat.begin(); it != det_stat.end(); ++it) {
    dFcnN[Desc(0, it->first.i, it->first.d)].fill(it->first.N, it->second);
    dFcnIt[Desc(it->first.N, 0, it->first.d)].fill(it->first.i, it->second);
  }

  map<Desc, RPGraphs> rFcnN;   // here Desc::N is 0 by definition
  map<Desc, RPGraphs> rFcnIt;  // here Desc::i is 0 by definition
  for (map<Desc, RPStat>::iterator it = rp_stat.begin(); it != rp_stat.end(); ++it) {
    rFcnN[Desc(0, it->first.i, it->first.d)].fill(it->first.N, it->second);
    rFcnIt[Desc(it->first.N, 0, it->first.d)].fill(it->first.i, it->second);
  }

  r = r.replace(r.find(':'), 1, ">");
  o = o.replace(o.find(':'), 1, ">");
  d = d.replace(d.find(':'), 1, ">");

  gDirectory = sf;
  gDirectory = mkdir(gDirectory, r.c_str());
  gDirectory = mkdir(gDirectory, o.c_str());
  gDirectory = mkdir(gDirectory, d.c_str());

  char buf[100];
  gDirectory = mkdir(gDirectory, "fcn_of_N");
  for (map<Desc, DetGraphs>::iterator it = dFcnN.begin(); it != dFcnN.end(); ++it) {
    sprintf(buf, "iteration>%u", it->first.i);
    gDirectory = mkdir(gDirectory, buf);
    sprintf(buf, "%u", it->first.d);
    gDirectory = mkdir(gDirectory, buf);
    it->second.write("tracks");
    gDirectory->cd("../..");
  }

  for (map<Desc, RPGraphs>::iterator it = rFcnN.begin(); it != rFcnN.end(); ++it) {
    sprintf(buf, "iteration>%u", it->first.i);
    gDirectory = mkdir(gDirectory, buf);
    sprintf(buf, "RP %u", it->first.d);
    gDirectory = mkdir(gDirectory, buf);
    it->second.write("tracks");
    gDirectory->cd("../..");
  }
  gDirectory->cd("..");

  gDirectory = mkdir(gDirectory, "fcn_of_iteration");
  for (map<Desc, DetGraphs>::iterator it = dFcnIt.begin(); it != dFcnIt.end(); ++it) {
    sprintf(buf, "N>%u", it->first.N);
    gDirectory = mkdir(gDirectory, buf);
    sprintf(buf, "%u", it->first.d);
    gDirectory = mkdir(gDirectory, buf);
    it->second.write("iteration");
    gDirectory->cd("../..");
  }

  for (map<Desc, RPGraphs>::iterator it = rFcnIt.begin(); it != rFcnIt.end(); ++it) {
    sprintf(buf, "N>%u", it->first.N);
    gDirectory = mkdir(gDirectory, buf);
    sprintf(buf, "RP %u", it->first.d);
    gDirectory = mkdir(gDirectory, buf);
    it->second.write("iteration");
    gDirectory->cd("../..");
  }
  gDirectory->cd("..");
}

//----------------------------------------------------------------------------------------------------

bool isRegDir(const dirent *de) {
  if (de->d_type != DT_DIR)
    return false;

  if (!strcmp(de->d_name, "."))
    return false;

  if (!strcmp(de->d_name, ".."))
    return false;

  return true;
}

//----------------------------------------------------------------------------------------------------

// Choices for RP input.
// * First of all, the Jan to Ideal comparison is not possible for RP data -
//   the factorizations are different, because of zero errors of the Ideal
//   algorithm.
// * PLAIN results do not contain RP data, do not use.
// * CUMULATIVE contain the factorization from the previous iteration.
// * FACTORED is probably the best choice.
string r_actual_file("./cumulative_factored_results_Jan.xml");
string r_ideal_file("./cumulative_factored_results_Ideal.xml");

// Choices for sensor input.
// * PLAIN results contain only the correction by the last iteration.
// * Expansion is not done, hence CUMULATIVE results contain internal ALIGNMENT
//   only. Factorization inherited from the previous iteration.
// * FACTORED results shall not be used - factorization is different for Jan and
//   Ideal algorithms (Ideal has zero errors).
// * If EXPANDED results are available, they might be used, they contain both
//   internal and RP alignment. Hence comparison between different misalignments
//   is not possible.
string s_actual_file("./cumulative_expanded_results_Jan.xml");
string s_ideal_file("./cumulative_expanded_results_Ideal.xml");

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentCorrectionsData LoadAlignment(const string &fn) {
  const auto &seq = CTPPSRPAlignmentCorrectionsMethods::loadFromXML(fn);
  if (seq.empty())
    throw cms::Exception("PPS") << "LoadAlignment: alignment sequence empty, file: " << fn;

  return seq[0].second;
}

//----------------------------------------------------------------------------------------------------

void PrintHelp(const char *name) {
  printf("USAGE: %s r_actual_file r_ideal_file s_actual_file s_ideal_file\n", name);
  printf("       %s --help (to print this help)\n", name);
  printf("PARAMETERS:\n");
  printf("\tr_actual_file\t file with actual RP results (default %s)\n", r_actual_file.c_str());
  printf("\tr_ideal_file\t file with ideal RP results (default %s)\n", r_ideal_file.c_str());
  printf("\ts_actual_file\t file with actual sensor results (default %s)\n", s_actual_file.c_str());
  printf("\ts_ideal_file\t file with ideal sensor results (default %s)\n", s_ideal_file.c_str());
}

//----------------------------------------------------------------------------------------------------

int main(int argc, const char *argv[]) {
  if (argc > 1) {
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
      PrintHelp(argv[0]);
      return 0;
    }

    if (argc > 1)
      r_actual_file = argv[1];
    if (argc > 2)
      r_ideal_file = argv[2];
    if (argc > 3)
      s_actual_file = argv[3];
    if (argc > 4)
      s_ideal_file = argv[4];
  }

  printf("r_actual_file: %s\n", r_actual_file.c_str());
  printf("r_ideal_file: %s\n", r_ideal_file.c_str());
  printf("s_actual_file: %s\n", s_actual_file.c_str());
  printf("s_ideal_file: %s\n", s_ideal_file.c_str());

  // open output file
  sf = new TFile("result_summary.root", "recreate");

  // traverse directory structure
  try {
    DIR *dp_r = opendir(".");
    dirent *de_r;
    // traverse RPs directories
    while ((de_r = readdir(dp_r))) {
      if (!isRegDir(de_r))
        continue;

      chdir(de_r->d_name);
      DIR *dp_o = opendir(".");
      // traverse optimized directories
      dirent *de_o;
      while ((de_o = readdir(dp_o))) {
        if (!isRegDir(de_o))
          continue;

        chdir(de_o->d_name);
        DIR *dp_d = opendir(".");
        dirent *de_d;
        // traverse tr_dist directories
        while ((de_d = readdir(dp_d))) {
          if (!isRegDir(de_d))
            continue;

          // sort the tr_N directories by N
          chdir(de_d->d_name);
          DIR *dp_N = opendir(".");
          dirent *de_N;
          map<unsigned int, string> nMap;
          while ((de_N = readdir(dp_N))) {
            if (isRegDir(de_N)) {
              string sN = de_N->d_name;
              sN = sN.substr(sN.find(':') + 1);
              unsigned int N = atof(sN.c_str());
              nMap[N] = de_N->d_name;
            }
          }

          resetStatistics();

          // traverse tr_N directories
          for (map<unsigned int, string>::iterator nit = nMap.begin(); nit != nMap.end(); ++nit) {
            chdir(nit->second.c_str());
            DIR *dp_i = opendir(".");
            dirent *de_i;
            // traverse repetitions directories
            while ((de_i = readdir(dp_i))) {
              if (!isRegDir(de_i))
                continue;

              chdir(de_i->d_name);

              // sort the iteration directories
              DIR *dp_it = opendir(".");
              dirent *de_it;
              map<unsigned int, string> itMap;
              while ((de_it = readdir(dp_it))) {
                if (isRegDir(de_it)) {
                  unsigned idx = 9;
                  if (de_it->d_name[9] == ':')
                    idx = 10;
                  unsigned int it = atoi(de_it->d_name + idx);
                  itMap[it] = de_it->d_name;
                }
              }

              // traverse iteration directories
              for (map<unsigned int, string>::iterator iit = itMap.begin(); iit != itMap.end(); ++iit) {
                chdir(iit->second.c_str());

                printf("%s|%s|%s|%s|%s|%s\n",
                       de_r->d_name,
                       de_o->d_name,
                       de_d->d_name,
                       nit->second.c_str(),
                       de_i->d_name,
                       iit->second.c_str());

                // load and process alignments
                try {
                  const auto &r_actual = LoadAlignment(r_actual_file);
                  const auto &r_ideal = LoadAlignment(r_ideal_file);
                  updateRPStatistics(nit->first, iit->first, r_actual, r_ideal);

                  const auto &s_actual = LoadAlignment(s_actual_file);
                  const auto &s_ideal = LoadAlignment(s_ideal_file);
                  updateSensorStatistics(nit->first, iit->first, s_actual, s_ideal);
                } catch (cms::Exception &e) {
                  printf("ERROR: A CMS exception has been caught:\n%s\nSkipping this directory.\n", e.what());
                }

                chdir("..");
              }

              closedir(dp_it);
              chdir("..");
            }

            closedir(dp_i);
            chdir("..");
          }

          writeGraphs(de_r->d_name, de_o->d_name, de_d->d_name);

          closedir(dp_N);
          chdir("..");
        }

        closedir(dp_d);
        chdir("..");
      }

      closedir(dp_o);
      chdir("..");
    }

    closedir(dp_r);
    chdir("..");

    delete sf;
  }

  catch (cms::Exception &e) {
    printf("ERROR: A CMS exception has been caught:\n%s\nStopping.\n", e.what());
  }

  catch (std::exception &e) {
    printf("ERROR: A std::exception has been caught:\n%s\nStopping.\n", e.what());
  }

  catch (...) {
    printf("ERROR: An exception has been caught, stopping.\n");
  }

  return 0;
}
