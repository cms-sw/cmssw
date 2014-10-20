#include <sstream>

#include "CondFormats/BTagObjects/interface/BTagEntry.h"

BTagEntry::Parameters::Parameters(
  OperatingPoint op, JetFlavor jf,
  std::string measurement_type, std::string sys_type,
  float eta_min, float eta_max, int reshaping_bin
):
  operatingPoint(op), jetFlavor(jf),
  measurementType(measurement_type), sysType(sys_type),
  etaMin(eta_min), etaMax(eta_max), reshapingBin(reshaping_bin)
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

BTagEntry::BTagEntry(const std::string &func,
                     BTagEntry::Parameters p,
                     float pt_min,
                     float pt_max):
  ptMin(pt_min),
  ptMax(pt_max),
  formula(func),
  params(p)
{}

BTagEntry::BTagEntry(const TF1* func, BTagEntry::Parameters p):
  params(p)
{
  double x1, x2;
  func->GetRange(x1, x2);  // needs doubles, not floats
  ptMin = x1;
  ptMax = x2;
  formula = std::string(func->GetExpFormula("p").Data());
}

// Creates step functions like this:
// "(bin_low_bound<=x&&x<bin_high_bound) ? bin_value : <next_bin>"
// e.g. "(1<=x&&x<2) ? 1 : (2<=x&&x<3) ? 2 : (3<=x&&x<4) ? 3 : 4"
BTagEntry::BTagEntry(const TH1* hist, BTagEntry::Parameters p):
  params(p)
{
  int nbins = hist->GetNbinsX();
  auto axis = hist->GetXaxis();
  ptMin = axis->GetBinLowEdge(1);
  ptMax = axis->GetBinUpEdge(nbins);

  std::stringstream buff;
  for (int i=1; i<nbins+1; ++i) {
    char tmp_buff[100];
    sprintf(tmp_buff,
            "(%g<=x&&x<%g) ? %g : ",  // %g is the smaller one of %e or %f
            axis->GetBinLowEdge(i),
            axis->GetBinUpEdge(i),
            hist->GetBinContent(i));
    buff << tmp_buff;
  }
  buff << 1.;  // default value
  formula = buff.str();
}
