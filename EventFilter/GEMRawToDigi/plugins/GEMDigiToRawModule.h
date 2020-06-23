#ifndef EventFilter_GEMRawToDigi_GEMDigiToRawModule_h
#define EventFilter_GEMRawToDigi_GEMDigiToRawModule_h

/** \class GEMDigiToRawModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"

namespace edm {
  class ConfigurationDescriptions;
}

class GEMDigiToRawModule : public edm::global::EDProducer<edm::RunCache<GEMROMapping> > {
public:
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMROMapping> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection> digi_token;
  edm::ESGetToken<GEMeMap, GEMeMapRcd> gemEMapToken_;
  bool useDBEMap_;
};
DEFINE_FWK_MODULE(GEMDigiToRawModule);
#endif
