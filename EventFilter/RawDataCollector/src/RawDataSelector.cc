//
// Original Author:  Marco ZANETTI
//         Created:  Mon Jan 28 18:22:13 CET 2008

#include <memory>
#include <utility>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/RawDataCollector/interface/RawDataFEDSelector.h"

class RawDataSelector : public edm::stream::EDProducer<> {
public:
  explicit RawDataSelector(const edm::ParameterSet&);

  ~RawDataSelector() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  RawDataFEDSelector* selector;

  edm::InputTag dataLabel;
  std::pair<int, int> fedRange;
};

RawDataSelector::RawDataSelector(const edm::ParameterSet& pset)
    : dataLabel(pset.getUntrackedParameter<edm::InputTag>("InputLabel", edm::InputTag("source"))) {
  fedRange = std::pair<int, int>(pset.getParameter<int>("lowerBound"), pset.getParameter<int>("upperBound"));

  selector = new RawDataFEDSelector();

  produces<FEDRawDataCollection>();
}

RawDataSelector::~RawDataSelector() { delete selector; }

void RawDataSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<FEDRawDataCollection> rawData;
  iEvent.getByLabel(dataLabel, rawData);

  /* here eventually perform some operation to get the list of FED's 
     to be written in the new collection.
     In this case we simply take the range from the ParameterSet */

  // the filtered raw data collections
  std::unique_ptr<FEDRawDataCollection> selectedRawData = selector->select(rawData, fedRange);

  iEvent.put(std::move(selectedRawData));
}

//define this as a plug-in
DEFINE_FWK_MODULE(RawDataSelector);
