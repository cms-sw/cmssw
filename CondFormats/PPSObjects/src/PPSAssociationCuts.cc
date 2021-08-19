/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Grzegorz Sroka
 ****************************************************************************/

#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <iostream>

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

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts::CutsPerArm &cutsPerArm) {
  os << "CutsPerArm {" << std::endl;

  os << "\tmeans {";
  for (auto const &value : cutsPerArm.means_) {
    os << "\"" << value << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\tthresholds {";
  for (auto const &value : cutsPerArm.thresholds_) {
    os << "\"" << value << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\ts_de_means {";
  for (auto const &value : cutsPerArm.s_de_means_) {
    os << "\"" << value->GetExpFormula() << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\ts_de_thresholds {";
  for (auto const &value : cutsPerArm.s_de_thresholds_) {
    os << "\"" << value->GetExpFormula() << "\", ";
  }
  os << "}" << std::endl << std::endl;

  os << "\tti_tr_min " << cutsPerArm.ti_tr_min_ << std::endl;
  os << "\tti_tr_max " << cutsPerArm.ti_tr_max_ << std::endl;
  os << "}" << std::endl;

  return os;
}

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts &ppsAssociationCuts) {
  os << "PPSAssociationCuts {" << std::endl;
  os << "45" << std::endl;
  os << ppsAssociationCuts.getAssociationCuts(0);
  os << "56" << std::endl;
  os << ppsAssociationCuts.getAssociationCuts(1);
  os << "}" << std::endl;

  return os;
}