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
#include <TF1.h>

#include <iostream>
#include <memory>
#include <cmath>

class PPSAssociationCuts {
public:
  class CutsPerArm {
  public:
    enum Quantities { qX, qY, qXi, qThetaY };

    double ti_tr_min_ = 0.;
    double ti_tr_max_ = 0.;

    std::vector<std::string> thresholds_;
    std::vector<std::string> means_;

    std::vector<std::shared_ptr<TF1> > s_de_means_ COND_TRANSIENT;
    std::vector<std::shared_ptr<TF1> > s_de_thresholds_ COND_TRANSIENT;

    bool isApplied(Quantities quantity) const {
      return (!thresholds_.at(quantity).empty()) && (!means_.at(quantity).empty());
    }

    bool isSatisfied(Quantities quantity, double x_near, double y_near, double xangle, double q_NF_diff) const {
      if (!isApplied(quantity))
        return true;
      const double mean = evaluateEquation(s_de_means_.at(quantity),x_near, y_near, xangle);
      const double threshold = evaluateEquation(s_de_thresholds_.at(quantity),x_near, y_near, xangle);
      return fabs(q_NF_diff - mean) < threshold;
    }

    CutsPerArm() {}

    ~CutsPerArm() {}

    CutsPerArm(const edm::ParameterSet &iConfig, int sector) {
      const auto &association_cuts = iConfig.getParameterSet("association_cuts_" + std::to_string(sector));

      const std::vector<std::string> names{"x", "y", "xi", "th_y"};
      for (std::size_t i = 0; i < names.size(); ++i) {
        std::string mean = association_cuts.getParameter<std::string>(names[i] + "_cut_mean");
        means_.push_back(mean);

        std::string threshold = association_cuts.getParameter<std::string>(names[i] + "_cut_threshold");
        thresholds_.push_back(threshold);

        s_de_means_.push_back(std::make_shared<TF1>("f", mean.c_str()));
        s_de_thresholds_.push_back(std::make_shared<TF1>("f", threshold.c_str()));
      }

      ti_tr_min_ = association_cuts.getParameter<double>("ti_tr_min");
      ti_tr_max_ = association_cuts.getParameter<double>("ti_tr_max");
    }

  private:
      double evaluateEquation(std::shared_ptr<TF1> equation,double x_near, double y_near, double xangle) const{
        equation->SetParameter("x_near",x_near);
        equation->SetParameter("y_near",y_near);
        equation->SetParameter("xangle",xangle);
        return equation->EvalPar(nullptr);
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
