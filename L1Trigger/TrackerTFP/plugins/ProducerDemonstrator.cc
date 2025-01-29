#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerDemonstrator
   *  \brief  ESProducer providing the algorithm to run input data through modelsim
   *          and to compares results with expected output data
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class ProducerDemonstrator : public ESProducer {
  public:
    ProducerDemonstrator(const ParameterSet& iConfig);
    ~ProducerDemonstrator() override {}
    unique_ptr<Demonstrator> produce(const DemonstratorRcd& rcd);

  private:
    Demonstrator::Config iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerDemonstrator::ProducerDemonstrator(const ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    iConfig_.dirIPBB_ = iConfig.getParameter<string>("DirIPBB");
    iConfig_.runTime_ = iConfig.getParameter<double>("RunTime");
    iConfig_.linkMappingIn_ = iConfig.getParameter<vector<int>>("LinkMappingIn");
    iConfig_.linkMappingOut_ = iConfig.getParameter<vector<int>>("LinkMappingOut");
  }

  unique_ptr<Demonstrator> ProducerDemonstrator::produce(const DemonstratorRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return make_unique<Demonstrator>(iConfig_, setup);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerDemonstrator);
