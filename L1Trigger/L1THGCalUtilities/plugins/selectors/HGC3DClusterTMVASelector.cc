#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"

#include "TMVA/Factory.h"
#include "TMVA/Reader.h"

namespace l1t {
  class HGC3DClusterTMVASelector : public edm::stream::EDProducer<> {
  public:
    explicit HGC3DClusterTMVASelector(const edm::ParameterSet &);
    ~HGC3DClusterTMVASelector() override {}

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

    edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> src_;
    StringCutObjectSelector<l1t::HGCalMulticluster> preselection_;
    std::vector<Var> variables_;
    std::string method_, weightsFile_;
    std::unique_ptr<TMVA::Reader> reader_;
    StringObjectFunction<l1t::HGCalMulticluster> wp_;

    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1t

l1t::HGC3DClusterTMVASelector::HGC3DClusterTMVASelector(const edm::ParameterSet &iConfig)
    : src_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      preselection_(iConfig.getParameter<std::string>("preselection")),
      method_(iConfig.getParameter<std::string>("method")),
      weightsFile_(iConfig.getParameter<std::string>("weightsFile")),
      reader_(new TMVA::Reader()),
      wp_(iConfig.getParameter<std::string>("wp")) {
  // first create all the variables
  for (const auto &psvar : iConfig.getParameter<std::vector<edm::ParameterSet>>("variables")) {
    variables_.emplace_back(psvar.getParameter<std::string>("name"), psvar.getParameter<std::string>("value"));
  }
  // then declare them
  for (auto &var : variables_)
    var.declare(*reader_);
  // then read the weights
  if (weightsFile_[0] != '/' && weightsFile_[0] != '.') {
    weightsFile_ = edm::FileInPath(weightsFile_).fullPath();
  }
  reco::details::loadTMVAWeights(&*reader_, method_, weightsFile_);
  // finally, declare outputs
  produces<l1t::HGCalMulticlusterBxCollection>();
  produces<l1t::HGCalMulticlusterBxCollection>("fail");
}

void l1t::HGC3DClusterTMVASelector::produce(edm::Event &iEvent, const edm::EventSetup &) {
  std::unique_ptr<l1t::HGCalMulticlusterBxCollection> out = std::make_unique<l1t::HGCalMulticlusterBxCollection>();
  std::unique_ptr<l1t::HGCalMulticlusterBxCollection> fail = std::make_unique<l1t::HGCalMulticlusterBxCollection>();

  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters;
  iEvent.getByToken(src_, multiclusters);

  for (int bx = multiclusters->getFirstBX(); bx <= multiclusters->getLastBX(); ++bx) {
    for (auto it = multiclusters->begin(bx), ed = multiclusters->end(bx); it != ed; ++it) {
      const auto &c = *it;
      if (preselection_(c)) {
        for (auto &var : variables_)
          var.fill(c);
        float mvaOut = reader_->EvaluateMVA(method_);
        if (mvaOut > wp_(c)) {
          out->push_back(bx, c);
        } else {
          fail->push_back(bx, c);
        }
      }
    }
  }

  iEvent.put(std::move(out));
  iEvent.put(std::move(fail), "fail");
}
using l1t::HGC3DClusterTMVASelector;
DEFINE_FWK_MODULE(HGC3DClusterTMVASelector);
