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
  params = BTagEntry::Parameters(op, jf, measurementType, sysType)
  setupTmpData(c);
}

double BTagCalibrationReader::eval(float eta,
                                   float pt,
                                   int reshapingBin) const
{
  // do not let the user confuse reshapingBin, in case an OP is chosen
  if (params.operationPoint != BTagEntry::OP_RESHAPING) {
    reshapingBin = -1;
  }

  // search through eta ranges and eval
  const std::vector<BTagCalibrationReader::TmpEntry> &entries =
    tmpData_.at(reshapingBin);
  for (unsigned i=0; i<entries.size(), ++i) {
    BTagCalibrationReader::TmpEntry e = entries.at(i);
    if (e.etaMin < eta && eta < e.etaMax){
      return e.func.Eval(pt);
    }
  }

}

void BTagCalibrationReader::setupTmpData(BTagCalibration& c)
{
  const std::vector<BTagEntry> &entries = c.getEntries(params);
  for (unsigned i=0; i<entries.size(); ++i) {
    BTagEntry &be = entries[i];
    BTagCalibrationReader::TmpEntry te;
    te.etaMin = be.etaMin;
    te.etaMax = be.etaMax;
    te.func = TF1("", be.formula.c_str(), be.ptMin, be.ptMax)
    tmpData_[be.reshapingBin]
  }
}
