#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"

void TriggerDQMBase::setMETitle(ObjME& me, const std::string& titleX, const std::string& titleY) {
  me.numerator->setAxisTitle(titleX, 1);
  me.numerator->setAxisTitle(titleY, 2);
  me.denominator->setAxisTitle(titleX, 1);
  me.denominator->setAxisTitle(titleY, 2);
}

void TriggerDQMBase::bookME(DQMStore::IBooker& ibooker, ObjME& me, const std::string& histname, const std::string& histtitle, const uint nbins, const double min, const double max) {
  me.numerator = ibooker.book1D(histname + "_numerator", histtitle + " (numerator)", nbins, min, max);
  me.denominator = ibooker.book1D(histname + "_denominator", histtitle + " (denominator)", nbins, min, max);
}

void TriggerDQMBase::bookME(DQMStore::IBooker& ibooker, ObjME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning) {
  uint nbins = binning.size() - 1;
  std::vector<float> fbinning(binning.begin(), binning.end());
  float* arr = &fbinning[0];
  me.numerator = ibooker.book1D(histname + "_numerator", histtitle + " (numerator)", nbins, arr);
  me.denominator = ibooker.book1D(histname + "_denominator", histtitle + " (denominator)", nbins, arr);
}

void TriggerDQMBase::bookME(DQMStore::IBooker& ibooker, ObjME& me, const std::string& histname, const std::string& histtitle, const uint nbinsX, const double xmin, const double xmax, const double ymin, const double ymax) {
  me.numerator = ibooker.bookProfile(histname + "_numerator", histtitle + " (numerator)", nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname + "_denominator", histtitle + " (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}

void TriggerDQMBase::bookME(DQMStore::IBooker& ibooker, ObjME& me, const std::string& histname, const std::string& histtitle, const uint nbinsX, const double xmin, const double xmax, const uint nbinsY, const double ymin, const double ymax) {
  me.numerator = ibooker.book2D(histname + "_numerator", histtitle + " (numerator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname + "_denominator", histtitle + " (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}

void TriggerDQMBase::bookME(DQMStore::IBooker& ibooker, ObjME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY){
  uint nbinsX = binningX.size() - 1;
  std::vector<float> fbinningX(binningX.begin(), binningX.end());
  float* arrX = &fbinningX[0];
  uint nbinsY = binningY.size() - 1;
  std::vector<float> fbinningY(binningY.begin(), binningY.end());
  float* arrY = &fbinningY[0];

  me.numerator = ibooker.book2D(histname + "_numerator", histtitle + " (numerator)", nbinsX, arrX, nbinsY, arrY);
  me.denominator = ibooker.book2D(histname + "_denominator", histtitle + " (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void TriggerDQMBase::fillHistoPSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<uint>("nbins");
  pset.add<double>("xmin");
  pset.add<double>("xmax");
}

void TriggerDQMBase::fillHistoLSPSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<uint>("nbins", 2500);
  pset.add<double>("xmin", 0.);
  pset.add<double>("xmax", 2500.);
}

TriggerDQMBase::MEbinning TriggerDQMBase::getHistoPSet(const edm::ParameterSet& pset) {
  return TriggerDQMBase::MEbinning{pset.getParameter<uint32_t>("nbins"), pset.getParameter<double>("xmin"), pset.getParameter<double>("xmax")};
}

TriggerDQMBase::MEbinning TriggerDQMBase::getHistoLSPSet(const edm::ParameterSet& pset) {
  return TriggerDQMBase::MEbinning{pset.getParameter<uint32_t>("nbins"), 0., double(pset.getParameter<uint32_t>("nbins"))};
}
