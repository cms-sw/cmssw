// F. Cossutti
//

// producer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <string>
#include <atomic>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace genFilterEff {
  struct Sums {
    mutable std::atomic<unsigned int> numEventsPassPos_ = {0};
    mutable std::atomic<unsigned int> numEventsPassNeg_ = {0};
    mutable std::atomic<unsigned int> numEventsTotalPos_ = {0};
    mutable std::atomic<unsigned int> numEventsTotalNeg_ = {0};
    mutable std::atomic<double> sumpass_w_ = {0};
    mutable std::atomic<double> sumpass_w2_ = {0};
    mutable std::atomic<double> sumtotal_w_ = {0};
    mutable std::atomic<double> sumtotal_w2_ = {0};
  };
}  // namespace genFilterEff

namespace {
  void atomic_sum_double(std::atomic<double>& oValue, double element) {
    double v = oValue.load();
    double sum = v + element;
    while (not oValue.compare_exchange_strong(v, sum)) {
      //some other thread updated oValue
      sum = v + element;
    }
  }
}  // namespace
using namespace genFilterEff;

class GenFilterEfficiencyProducer
    : public edm::global::EDProducer<edm::EndLuminosityBlockProducer, edm::LuminosityBlockCache<Sums>> {
public:
  explicit GenFilterEfficiencyProducer(const edm::ParameterSet&);
  ~GenFilterEfficiencyProducer() override;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override;
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, const edm::EventSetup&) const override;

  std::shared_ptr<Sums> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override;
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;

  std::string filterPath;

  edm::service::TriggerNamesService* tns_;

  std::string thisProcess;
  unsigned int pathIndex;
};

GenFilterEfficiencyProducer::GenFilterEfficiencyProducer(const edm::ParameterSet& iConfig)
    : filterPath(iConfig.getParameter<std::string>("filterPath")), tns_(), thisProcess(), pathIndex(100000) {
  //now do what ever initialization is needed
  if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
    // get tns pointer
    tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
    if (tns_ != nullptr) {
      thisProcess = tns_->getProcessName();
      std::vector<std::string> theNames = tns_->getTrigPaths();
      for (unsigned int i = 0; i < theNames.size(); i++) {
        if (theNames[i] == filterPath) {
          pathIndex = i;
          continue;
        }
      }
    } else
      edm::LogError("ServiceNotAvailable") << "TriggerNamesServive not available, no filter information stored";
  }

  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", thisProcess));
  genEventInfoToken_ = consumes<GenEventInfoProduct>(edm::InputTag("generator", ""));
  produces<GenFilterInfo, edm::Transition::EndLuminosityBlock>();
}

GenFilterEfficiencyProducer::~GenFilterEfficiencyProducer() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void GenFilterEfficiencyProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<edm::TriggerResults> trigR;
  iEvent.getByToken(triggerResultsToken_, trigR);
  edm::Handle<GenEventInfoProduct> genEventScale;
  iEvent.getByToken(genEventInfoToken_, genEventScale);
  if (!genEventScale.isValid())
    return;
  double weight = genEventScale->weight();

  auto sums = luminosityBlockCache(iEvent.getLuminosityBlock().index());

  unsigned int nSize = (*trigR).size();
  // std::cout << "Number of paths in TriggerResults = " << nSize  << std::endl;
  if (nSize >= pathIndex) {
    if (!trigR->wasrun(pathIndex))
      return;
    if (trigR->accept(pathIndex)) {
      atomic_sum_double(sums->sumpass_w_, weight);
      atomic_sum_double(sums->sumpass_w2_, weight * weight);

      atomic_sum_double(sums->sumtotal_w_, weight);
      atomic_sum_double(sums->sumtotal_w2_, weight * weight);

      if (weight > 0) {
        sums->numEventsPassPos_++;
        sums->numEventsTotalPos_++;
      } else {
        sums->numEventsPassNeg_++;
        sums->numEventsTotalNeg_++;
      }

    } else  // if fail the filter
    {
      atomic_sum_double(sums->sumtotal_w_, weight);
      atomic_sum_double(sums->sumtotal_w2_, weight * weight);

      if (weight > 0)
        sums->numEventsTotalPos_++;
      else
        sums->numEventsTotalNeg_++;
    }
    //    std::cout << "Total events = " << numEventsTotal << " passed = " << numEventsPassed << std::endl;
  }
}

std::shared_ptr<Sums> GenFilterEfficiencyProducer::globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                              edm::EventSetup const&) const {
  return std::make_shared<Sums>();
}

void GenFilterEfficiencyProducer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                           const edm::EventSetup&) const {}

void GenFilterEfficiencyProducer::globalEndLuminosityBlockProduce(edm::LuminosityBlock& iLumi,
                                                                  const edm::EventSetup&) const {
  auto sums = luminosityBlockCache(iLumi.index());
  std::unique_ptr<GenFilterInfo> thisProduct(new GenFilterInfo(sums->numEventsPassPos_,
                                                               sums->numEventsPassNeg_,
                                                               sums->numEventsTotalPos_,
                                                               sums->numEventsTotalNeg_,
                                                               sums->sumpass_w_,
                                                               sums->sumpass_w2_,
                                                               sums->sumtotal_w_,
                                                               sums->sumtotal_w2_));
  iLumi.put(std::move(thisProduct));
}

DEFINE_FWK_MODULE(GenFilterEfficiencyProducer);
