#include "CondFormats/BTagObjects/interface/BTagCalibrationReader.h"

BTagCalibrationReader::BTagCalibrationReader(BTagCalibration& c,
                                             BTagEntry::Parameters p):
  params(p)
{
  setupTmpData(c);
}

BTagCalibrationReader::BTagCalibrationReader(BTagCalibration& c,
                                             BTagEntry::OperatingPoint op,
                                             BTagEntry::JetFlavor jf,
                                             std::string measurementType,
                                             std::string sysType)
{
  params = BTagEntry::Parameters(op, jf, measurementType, sysType);
  setupTmpData(c);
}

double BTagCalibrationReader::eval(float eta,
                                   float pt,
                                   float discr) const
{
  bool use_discr = (params.operatingPoint == BTagEntry::OP_RESHAPING);

  // search linearly through eta, pt and discr ranges and eval
  // future: find some clever data structure based on intervals
  for (unsigned i=0; i<tmpData_.size(); ++i) {
    const BTagCalibrationReader::TmpEntry &e = tmpData_.at(i);
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

  return 1.;  // default value
}

void BTagCalibrationReader::setupTmpData(BTagCalibration& c)
{
  const std::vector<BTagEntry> &entries = c.getEntries(params);
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

    tmpData_.push_back(te);
  }
}
