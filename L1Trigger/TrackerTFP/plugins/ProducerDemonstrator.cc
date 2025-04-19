#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <vector>
#include <string>
#include <memory>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerDemonstrator
   *  \brief  ESProducer providing the algorithm to run input data through modelsim
   *          and to compares results with expected output data
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class ProducerDemonstrator : public edm::ESProducer {
  public:
    ProducerDemonstrator(const edm::ParameterSet& iConfig);
    ~ProducerDemonstrator() override {}
    std::unique_ptr<Demonstrator> produce(const tt::SetupRcd& rcd);

  private:
    Demonstrator::Config iConfig_;
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  };

  ProducerDemonstrator::ProducerDemonstrator(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    iConfig_.dirIPBB_ = iConfig.getParameter<std::string>("DirIPBB");
    iConfig_.runTime_ = iConfig.getParameter<double>("RunTime");
    iConfig_.linkMappingIn_ = iConfig.getParameter<std::vector<int>>("LinkMappingIn");
    iConfig_.linkMappingOut_ = iConfig.getParameter<std::vector<int>>("LinkMappingOut");
  }

  std::unique_ptr<Demonstrator> ProducerDemonstrator::produce(const tt::SetupRcd& rcd) {
    const tt::Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<Demonstrator>(iConfig_, setup);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerDemonstrator);
