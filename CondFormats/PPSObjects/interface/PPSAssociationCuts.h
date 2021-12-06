/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Grzegorz Sroka
 ****************************************************************************/

#ifndef CondFormats_PPSObjects_PPSAssociationCuts_h
#define CondFormats_PPSObjects_PPSAssociationCuts_h

struct TF1;

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <memory>
#include <cmath>
#include <vector>

class PPSAssociationCuts {
public:
  class CutsPerArm {
  public:
    enum Quantities { qX, qY, qXi, qThetaY };

    CutsPerArm() {}

    CutsPerArm(const edm::ParameterSet &iConfig, int sector);

    ~CutsPerArm() {}

    const std::vector<std::string> &getMeans() const { return s_means_; }
    const std::vector<std::string> &getThresholds() const { return s_thresholds_; }

    double getTiTrMin() const { return ti_tr_min_; }
    double getTiTrMax() const { return ti_tr_max_; }

    // build TF1 representations of the mean and threshold functions
    void buildFunctions();

    // returns whether the specified cut is applied
    bool isApplied(Quantities quantity) const;

    // returns whether if the specified cut is satisfied (for a particular event)
    bool isSatisfied(Quantities quantity, double x_near, double y_near, double xangle, double q_NF_diff) const;

  protected:
    // string representation of the cut parameters - for serialisation
    std::vector<std::string> s_means_;
    std::vector<std::string> s_thresholds_;

    // TF1 representation of the cut parameters - for run time evaluations
    std::vector<std::shared_ptr<TF1> > f_means_ COND_TRANSIENT;
    std::vector<std::shared_ptr<TF1> > f_thresholds_ COND_TRANSIENT;

    // timing-tracking cuts
    double ti_tr_min_;
    double ti_tr_max_;

    static double evaluateExpression(std::shared_ptr<TF1> expression, double x_near, double y_near, double xangle);

    COND_SERIALIZABLE;
  };

  PPSAssociationCuts() {}

  PPSAssociationCuts(const edm::ParameterSet &iConfig);

  ~PPSAssociationCuts() {}

  // checks if the data have a valid structure
  bool isValid() const;

  // builds run-time data members, useful e.g. after loading data from DB
  void initialize();

  const CutsPerArm &getAssociationCuts(const int sector) const { return association_cuts_.find(sector)->second; }

  static edm::ParameterSetDescription getDefaultParameters();

private:
  std::map<unsigned int, CutsPerArm> association_cuts_;

  COND_SERIALIZABLE;
};

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts::CutsPerArm &cutsPerArm);

std::ostream &operator<<(std::ostream &os, const PPSAssociationCuts &ppsAssociationCuts);

#endif
