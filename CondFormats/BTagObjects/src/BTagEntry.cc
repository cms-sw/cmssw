#include <sstream>

#include "CondFormats/BTagObjects/interface/BTagEntry.h"

BTagEntry::Parameters::Parameters(
  OperatingPoint op, JetFlavor jf,
  std::string measurement_type, std::string sys_type,
  float eta_min, float eta_max,
  float pt_min, float pt_max,
  float discr_min, float discr_max
):
  operatingPoint(op), jetFlavor(jf),
  measurementType(measurement_type), sysType(sys_type),
  etaMin(eta_min), etaMax(eta_max),
  ptMin(pt_min), ptMax(pt_max),
  discrMin(discr_min), discrMax(discr_max)
{}

std::string BTagEntry::Parameters::token()
{
  std::stringstream buff;
  buff << operatingPoint << ", "
       << jetFlavor << ", "
       << measurementType << ", "
       << sysType;
  return buff.str();
}

BTagEntry::BTagEntry(const std::string &func, BTagEntry::Parameters p):
  formula(func),
  params(p)
{}

BTagEntry::BTagEntry(const TF1* func, BTagEntry::Parameters p):
  formula(std::string(func->GetExpFormula("p").Data())),
  params(p)
{}

// Creates chained step functions like this:
// "<prevous_bin> : x<bin_high_bound ? bin_value : <next_bin>"
// e.g. "x<0 ? 1 : x<1 ? 2 : x<2 ? 3 : 4"
BTagEntry::BTagEntry(const TH1* hist, BTagEntry::Parameters p):
  params(p)
{
  int nbins = hist->GetNbinsX();
  auto axis = hist->GetXaxis();

  std::stringstream buff;
  buff << "x<" << axis->GetBinLowEdge(1) << " ? 0. : ";  // default value
  for (int i=1; i<nbins+1; ++i) {
    char tmp_buff[100];
    sprintf(tmp_buff,
            "x<%g ? %g : ",  // %g is the smaller one of %e or %f
            axis->GetBinUpEdge(i),
            hist->GetBinContent(i));
    buff << tmp_buff;
  }
  buff << 0.;  // default value
  formula = buff.str();
}
