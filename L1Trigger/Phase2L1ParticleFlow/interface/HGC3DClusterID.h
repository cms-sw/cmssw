#ifndef L1Trigger_Phase2L1ParticleFlow_HGC3DClusterID_h
#define L1Trigger_Phase2L1ParticleFlow_HGC3DClusterID_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
//#include "conifer.h"
#include "conifer_cpp.h"

#include <vector>
#include <cmath>
#include <algorithm>

// For conifer model inference
typedef ap_fixed<20, 10> bdt_feature_t;
typedef ap_fixed<20, 6> bdt_score_t;

conifer::BDT<bdt_feature_t, bdt_score_t, false> *multiclass_bdt_;

std::vector<float> wp_PU;
std::vector<float> wp_Pi;
std::vector<float> wp_Eg;

std::vector<bdt_feature_t> inputs;
std::vector<bdt_score_t> bdt_score;

namespace l1tpf {
  class HGC3DClusterID {
  public:
    HGC3DClusterID(const edm::ParameterSet &pset);

    float evaluate(const l1t::HGCalMulticluster &cl, l1t::PFCluster &cpf);

    bool passPuID(l1t::PFCluster &cpf, float maxScore);
    bool passPFEmID(l1t::PFCluster &cpf, float maxScore);
    bool passEgEmID(l1t::PFCluster &cpf, float maxScore);
    bool passPiID(l1t::PFCluster &cpf, float maxScore);

  private:
    class Var {
    public:
      Var(const std::string &name, const std::string &expr) : name_(name), expr_(expr) {}
      // void declare(TMVA::Reader &r) { r.AddVariable(name_, &val_); }
      void fill(const l1t::HGCalMulticluster &c) { val_ = expr_(c); }

    private:
      std::string name_;
      StringObjectFunction<l1t::HGCalMulticluster> expr_;
      float val_;
    };

    std::vector<Var> variables_;


  };  //class
};    // namespace l1tpf

#endif
