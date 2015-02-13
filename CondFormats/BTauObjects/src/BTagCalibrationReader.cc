#include "CondFormats/BTauObjects/interface/BTagCalibrationReader.h"

BTagCalibrationReader::BTagCalibrationReader(const BTagCalibration* c,
                                             BTagEntry::OperatingPoint op,
                                             std::string measurementType,
                                             std::string sysType):
  params(BTagEntry::Parameters(op, measurementType, sysType)),
  useAbsEta(true)
{
  setupTmpData(c);
}

double BTagCalibrationReader::eval(BTagEntry::JetFlavor jf,
                                   float eta,
                                   float pt,
                                   float discr) const
{
  bool use_discr = (params.operatingPoint == BTagEntry::OP_RESHAPING);
  if (useAbsEta[jf] && eta < 0) {
    eta = -eta;
  }

  // search linearly through eta, pt and discr ranges and eval
  // future: find some clever data structure based on intervals
  const std::vector<TmpEntry> &entries = tmpData_.at(jf);
  for (unsigned i=0; i<entries.size(); ++i) {
    const BTagCalibrationReader::TmpEntry &e = entries.at(i);
    if (
      e.etaMin <= eta && eta < e.etaMax                   // find eta
      && e.ptMin <= pt && pt < e.ptMax                    // check pt
    ){
      if (use_discr) {                                    // discr. reshaping?
        if (e.discrMin <= discr && discr < e.discrMax) {  // check discr
          return e.func.Eval(discr);
        }
      } else {
        return e.func.Eval(pt);
      }
    }
  }

  return 0.;  // default value
}

void BTagCalibrationReader::setupTmpData(const BTagCalibration* c)
{
  useAbsEta = std::vector<bool>(4, true);
  const std::vector<BTagEntry> &entries = c->getEntries(params);
  for (unsigned i=0; i<entries.size(); ++i) {
    const BTagEntry &be = entries[i];
    BTagCalibrationReader::TmpEntry te;
    te.etaMin = be.params.etaMin;
    te.etaMax = be.params.etaMax;
    te.ptMin = be.params.ptMin;
    te.ptMax = be.params.ptMax;
    te.discrMin = be.params.discrMin;
    te.discrMax = be.params.discrMax;

    if (params.operatingPoint == BTagEntry::OP_RESHAPING) {
      te.func = TF1("", be.formula.c_str(),
                    be.params.discrMin, be.params.discrMax);
    } else {
      te.func = TF1("", be.formula.c_str(),
                    be.params.ptMin, be.params.ptMax);
    }

    tmpData_[be.params.jetFlavor].push_back(te);
    if (te.etaMin < 0) {
      useAbsEta[be.params.jetFlavor] = false;
    }
  }
}
