#ifndef L1Trigger_Phase2L1ParticleFlow_HGC3DClusterEgID_h
#define L1Trigger_Phase2L1ParticleFlow_HGC3DClusterEgID_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "TMVA/Factory.h"
#include "TMVA/Reader.h"

#include <vector>
#include <cmath>

namespace l1tpf {
  class HGC3DClusterEgID {
  public:
    HGC3DClusterEgID(const edm::ParameterSet &pset);

    void prepareTMVA();

    float passID(l1t::HGCalMulticluster c, l1t::PFCluster &cpf);

    std::string method() { return method_; }

  private:
    class Var {
    public:
      Var(const std::string &name, const std::string &expr) : name_(name), expr_(expr) {}
      void declare(TMVA::Reader &r) { r.AddVariable(name_, &val_); }
      void fill(const l1t::HGCalMulticluster &c) { val_ = expr_(c); }

    private:
      std::string name_;
      StringObjectFunction<l1t::HGCalMulticluster> expr_;
      float val_;
    };

    bool isPUFilter_;
    StringCutObjectSelector<l1t::HGCalMulticluster> preselection_;
    std::vector<Var> variables_;
    std::string method_, weightsFile_;
    std::unique_ptr<TMVA::Reader> reader_;
    StringObjectFunction<l1t::HGCalMulticluster> wp_;
  };  //class
};    // namespace l1tpf

#endif
