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

  desc.add<bool>("x_cut_apply", false)->setComment("whether to apply track-association cut in x");
  desc.add<double>("x_cut_mean", 0E-6)->setComment("mean of track-association cut in x, mm");
  desc.add<double>("x_cut_value", 800E-6)->setComment("threshold of track-association cut in x, mm");

  desc.add<bool>("y_cut_apply", false)->setComment("whether to apply track-association cut in y");
  desc.add<double>("y_cut_mean", 0E-6)->setComment("mean of track-association cut in y, mm");
  desc.add<double>("y_cut_value", 600E-6)->setComment("threshold of track-association cut in y, mm");

  desc.add<bool>("xi_cut_apply", true)->setComment("whether to apply track-association cut in xi");
  desc.add<double>("xi_cut_mean", 0.)->setComment("mean of track-association cut in xi");
  desc.add<double>("xi_cut_value", 0.013)->setComment("threshold of track-association cut in xi");

  desc.add<bool>("th_y_cut_apply", true)->setComment("whether to apply track-association cut in th_y");
  desc.add<double>("th_y_cut_mean", 0E-6)->setComment("mean of track-association cut in th_y, rad");
  desc.add<double>("th_y_cut_value", 20E-6)->setComment("threshold of track-association cut in th_y, rad");

  desc.add<double>("ti_tr_min", -1.)->setComment("minimum value for timing-tracking association cut");
  desc.add<double>("ti_tr_max", +1.)->setComment("maximum value for timing-tracking association cut");

  return desc;
}

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts::CutsPerArm &cutsPerArm) {
  os << "CutsPerArm {" << std::endl;
  os << "\tx_cut_apply " << cutsPerArm.x_cut_apply << std::endl;
  os << "\tx_cut_value " << cutsPerArm.x_cut_value << std::endl;
  os << "\tx_cut_mean " << cutsPerArm.x_cut_mean << std::endl;
  os << std::endl;
  os << "\ty_cut_apply " << cutsPerArm.y_cut_apply << std::endl;
  os << "\ty_cut_value " << cutsPerArm.y_cut_value << std::endl;
  os << "\ty_cut_mean " << cutsPerArm.y_cut_mean << std::endl;
  os << std::endl;
  os << "\txi_cut_apply " << cutsPerArm.xi_cut_apply << std::endl;
  os << "\txi_cut_value " << cutsPerArm.xi_cut_value << std::endl;
  os << "\txi_cut_mean " << cutsPerArm.xi_cut_mean << std::endl;
  os << std::endl;
  os << "\tth_y_cut_apply " << cutsPerArm.th_y_cut_apply << std::endl;
  os << "\tth_y_cut_value " << cutsPerArm.th_y_cut_value << std::endl;
  os << "\tth_y_cut_mean " << cutsPerArm.th_y_cut_mean << std::endl;
  os << std::endl;
  os << "\tti_tr_min " << cutsPerArm.ti_tr_min << std::endl;
  os << "\tti_tr_max " << cutsPerArm.ti_tr_max << std::endl;
  os << "\t}" << std::endl;

  return os;
}

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts &ppsAssociationCuts) {
  os << "PPSAssociationCuts {" << std::endl;
  os << "45" << std::endl;
  os << "\t" << ppsAssociationCuts.getAssociationCuts(0);
  os << "56" << std::endl;
  os << "\t" << ppsAssociationCuts.getAssociationCuts(1);
  os << "}" << std::endl;

  return os;
}