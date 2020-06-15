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
#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"

#include "TMVA/Factory.h"
#include "TMVA/Reader.h"

#include <vector>
#include <cmath>

namespace l1tpf {
  class HGC3DClusterEgID {
  public:
    HGC3DClusterEgID(const edm::ParameterSet &pset)
        : isPUFilter_(pset.getParameter<bool>("isPUFilter")),
          preselection_(pset.getParameter<std::string>("preselection")),
          method_(pset.getParameter<std::string>("method")),
          weightsFile_(pset.getParameter<std::string>("weightsFile")),
          reader_(new TMVA::Reader()),
          wp_(pset.getParameter<std::string>("wp")) {
      // first create all the variables
      for (const auto &psvar : pset.getParameter<std::vector<edm::ParameterSet>>("variables")) {
        variables_.emplace_back(psvar.getParameter<std::string>("name"), psvar.getParameter<std::string>("value"));
      }
    }

    void prepareTMVA() {
      // Declare the variables
      for (auto &var : variables_)
        var.declare(*reader_);
      // then read the weights
      if (weightsFile_[0] != '/' && weightsFile_[0] != '.') {
        weightsFile_ = edm::FileInPath(weightsFile_).fullPath();
      }
      reco::details::loadTMVAWeights(&*reader_, method_, weightsFile_);
    }

    float passID(l1t::HGCalMulticluster c, l1t::PFCluster &cpf) {
      if (preselection_(c)) {
        for (auto &var : variables_)
          var.fill(c);
        float mvaOut = reader_->EvaluateMVA(method_);
        if (isPUFilter_)
          cpf.setEgVsPUMVAOut(mvaOut);
        else
          cpf.setEgVsPionMVAOut(mvaOut);
        return (mvaOut > wp_(c) ? 1 : 0);
      } else {
        if (isPUFilter_)
          cpf.setEgVsPUMVAOut(-100.0);
        else
          cpf.setEgVsPionMVAOut(-100.0);
        return 0;
      }
    }

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
