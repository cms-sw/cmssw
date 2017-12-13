#ifndef EventFilter_GEMRawToDigi_GEMDigiToRawModule_h
#define EventFilter_GEMRawToDigi_GEMDigiToRawModule_h

/** \class GEMDigiToRawModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/GEMObjects/interface/GEMROmap.h"

namespace edm {
  class ConfigurationDescriptions;
}

class GEMDigiToRawModule : public edm::stream::EDProducer<> {
 public:
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet & pset);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  // Operations
  void produce(edm::Event&, edm::EventSetup const&) override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection> digi_token;
  bool useDBEMap_;

  const GEMEMap* m_gemEMap;
  const GEMROmap* m_gemROMap;
  
};
DEFINE_FWK_MODULE(GEMDigiToRawModule);
#endif


