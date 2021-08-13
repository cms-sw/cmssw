/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Grzegorz Sroka
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_PPSAssociationCuts_h
#define CondFormats_PPSObjects_PPSAssociationCuts_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>

class PPSAssociationCuts {
public:
  class CutsPerArm {
  public:
    bool x_cut_apply = false;
    double x_cut_value = 0.;
    double x_cut_mean = 0.;
    bool y_cut_apply = false;
    double y_cut_value = 0.;
    double y_cut_mean = 0.;
    bool xi_cut_apply = false;
    double xi_cut_value = 0.;
    double xi_cut_mean = 0.;
    bool th_y_cut_apply = false;
    double th_y_cut_value = 0.;
    double th_y_cut_mean = 0.;
    double ti_tr_min = 0.;
    double ti_tr_max = 0.;

    CutsPerArm() {}

    CutsPerArm(const edm::ParameterSet &iConfig, int sector) {
      const auto &association_cuts = iConfig.getParameterSet("association_cuts_" + std::to_string(sector));
      x_cut_apply = association_cuts.getParameter<bool>("x_cut_apply");
      x_cut_value = association_cuts.getParameter<double>("x_cut_value");
      x_cut_mean = association_cuts.getParameter<double>("x_cut_mean");

      y_cut_apply = association_cuts.getParameter<bool>("y_cut_apply");
      y_cut_value = association_cuts.getParameter<double>("y_cut_value");
      y_cut_mean = association_cuts.getParameter<double>("y_cut_mean");

      xi_cut_apply = association_cuts.getParameter<bool>("xi_cut_apply");
      xi_cut_value = association_cuts.getParameter<double>("xi_cut_value");
      xi_cut_mean = association_cuts.getParameter<double>("xi_cut_mean");

      th_y_cut_apply = association_cuts.getParameter<bool>("th_y_cut_apply");
      th_y_cut_value = association_cuts.getParameter<double>("th_y_cut_value");
      th_y_cut_mean = association_cuts.getParameter<double>("th_y_cut_mean");

      ti_tr_min = association_cuts.getParameter<double>("ti_tr_min");
      ti_tr_max = association_cuts.getParameter<double>("ti_tr_max");
    }
    COND_SERIALIZABLE;
  };

  PPSAssociationCuts(const edm::ParameterSet &iConfig) {
    int i = 0;
    for (const int &sector : {45, 56}) {
      association_cuts_[i++] = CutsPerArm(iConfig, sector);
    }
  }

  PPSAssociationCuts() {}
  ~PPSAssociationCuts() {}

  const CutsPerArm &getAssociationCuts(const int sector) const { return association_cuts_.at(sector); }

  static edm::ParameterSetDescription getDefaultParameters();

private:
  std::map<unsigned int, CutsPerArm> association_cuts_;

  COND_SERIALIZABLE;
};

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts::CutsPerArm &cutsPerArm);

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts &ppsAssociationCuts);

#endif
