#include "CondFormats/BTagObjects/interface/BTagEntry.h"

BTagEntry::Parameters::Parameters(
  OperatingPoint op, JetFlavor jf,
  std::string measurement_type, std::string sys_type,
  float etaMin, float etaMax, int reshaping_bin
):
  operationPoint(op), jetFlavor(jf),
  measurementType(measurement_type), sysType(sys_type),
  etaMin(eta_min), etaMax(eta_max), reshapingBin(reshaping_bin)
{}

std::string BTagEntry::Parameters::token()
{
  char buff[100];
  sprintf(buff, "%d, %d, %s, %s",
          operationPoint, jetFlavor, measurementType, sysType);
  return std::string(buff);
}

BTagEntry::BTagEntry(const std::string &func,
                     BTagEntry::Parameters p
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
  func->GetRange(ptMin, ptMax)
  formula = std::string(func->GetExpFormula("p").Data())
}

// TODO histo to stepfunction constructor