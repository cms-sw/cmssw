/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Grzegorz Sroka
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "TF1.h"

#include <iostream>

//----------------------------------------------------------------------------------------------------

PPSAssociationCuts::CutsPerArm::CutsPerArm(const edm::ParameterSet &iConfig, int sector) {
  const auto &association_cuts = iConfig.getParameterSet("association_cuts_" + std::to_string(sector));

  const std::vector<std::string> names{"x", "y", "xi", "th_y"};
  for (std::size_t i = 0; i < names.size(); ++i) {
    std::string mean = association_cuts.getParameter<std::string>(names[i] + "_cut_mean");
    s_means_.push_back(mean);

    std::string threshold = association_cuts.getParameter<std::string>(names[i] + "_cut_threshold");
    s_thresholds_.push_back(threshold);
  }

  ti_tr_min_ = association_cuts.getParameter<double>("ti_tr_min");
  ti_tr_max_ = association_cuts.getParameter<double>("ti_tr_max");
}

//----------------------------------------------------------------------------------------------------

void PPSAssociationCuts::CutsPerArm::buildFunctions() {
  f_means_.clear();
  for (const auto &s : s_means_)
    f_means_.push_back(std::make_shared<TF1>("f", s.c_str()));

  f_thresholds_.clear();
  for (const auto &s : s_thresholds_)
    f_thresholds_.push_back(std::make_shared<TF1>("f", s.c_str()));
}

//----------------------------------------------------------------------------------------------------

bool PPSAssociationCuts::CutsPerArm::isApplied(Quantities quantity) const {
  return (!s_thresholds_[quantity].empty()) && (!s_means_[quantity].empty());
}

//----------------------------------------------------------------------------------------------------

bool PPSAssociationCuts::CutsPerArm::isSatisfied(
    Quantities quantity, double x_near, double y_near, double xangle, double q_NF_diff) const {
  // if cut not applied, then condition considered as satisfied
  if (!isApplied(quantity))
    return true;

  // evaluate mean and threshold
  const double mean = evaluateExpression(f_means_[quantity], x_near, y_near, xangle);
  const double threshold = evaluateExpression(f_thresholds_[quantity], x_near, y_near, xangle);

  // make comparison
  return fabs(q_NF_diff - mean) < threshold;
}

//----------------------------------------------------------------------------------------------------

double PPSAssociationCuts::CutsPerArm::evaluateExpression(std::shared_ptr<TF1> expression,
                                                          double x_near,
                                                          double y_near,
                                                          double xangle) {
  expression->SetParameter("x_near", x_near);
  expression->SetParameter("y_near", y_near);
  expression->SetParameter("xangle", xangle);
  return expression->EvalPar(nullptr);
}

//----------------------------------------------------------------------------------------------------

PPSAssociationCuts::PPSAssociationCuts(const edm::ParameterSet &iConfig) {
  unsigned int i = 0;
  for (const int &sector : {45, 56})
    association_cuts_[i++] = CutsPerArm(iConfig, sector);

  initialize();
}

//----------------------------------------------------------------------------------------------------

edm::ParameterSetDescription PPSAssociationCuts::getDefaultParameters() {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("x_cut_mean", "")->setComment("mean of track-association cut in x, mm");
  desc.add<std::string>("x_cut_threshold", "")->setComment("threshold of track-association cut in x, mm");

  desc.add<std::string>("y_cut_mean", "")->setComment("mean of track-association cut in y, mm");
  desc.add<std::string>("y_cut_threshold", "")->setComment("threshold of track-association cut in y, mm");

  desc.add<std::string>("xi_cut_mean", "")->setComment("mean of track-association cut in xi");
  desc.add<std::string>("xi_cut_threshold", "")->setComment("threshold of track-association cut in xi");

  desc.add<std::string>("th_y_cut_mean", "")->setComment("mean of track-association cut in th_y, rad");
  desc.add<std::string>("th_y_cut_threshold", "")->setComment("threshold of track-association cut in th_y, rad");

  desc.add<double>("ti_tr_min", -1.)->setComment("minimum value for timing-tracking association cut");
  desc.add<double>("ti_tr_max", +1.)->setComment("maximum value for timing-tracking association cut");

  return desc;
}

//----------------------------------------------------------------------------------------------------

bool PPSAssociationCuts::isValid() const {
  // valid if association_cuts_ have two entries, with keys 0 and 1
  if (association_cuts_.size() != 2)
    return false;

  if (association_cuts_.find(0) == association_cuts_.end())
    return false;
  if (association_cuts_.find(1) == association_cuts_.end())
    return false;

  return true;
}

//----------------------------------------------------------------------------------------------------

void PPSAssociationCuts::initialize() {
  if (!isValid())
    throw cms::Exception("PPS") << "Invalid structure of PPSAssociationCuts.";

  for (auto &p : association_cuts_)
    p.second.buildFunctions();
}

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts::CutsPerArm &cutsPerArm) {
  os << "CutsPerArm {" << std::endl;

  os << "\tmeans {";
  for (auto const &value : cutsPerArm.getMeans()) {
    os << "\"" << value << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\tthresholds {";
  for (auto const &value : cutsPerArm.getThresholds()) {
    os << "\"" << value << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\tti_tr_min " << cutsPerArm.getTiTrMin() << std::endl;
  os << "\tti_tr_max " << cutsPerArm.getTiTrMax() << std::endl;
  os << "}" << std::endl;

  return os;
}

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts &ppsAssociationCuts) {
  os << "PPSAssociationCuts {" << std::endl;
  os << "45" << std::endl;
  os << ppsAssociationCuts.getAssociationCuts(0);
  os << "56" << std::endl;
  os << ppsAssociationCuts.getAssociationCuts(1);
  os << "}" << std::endl;

  return os;
}