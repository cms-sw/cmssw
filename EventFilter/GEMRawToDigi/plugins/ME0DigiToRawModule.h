#ifndef EventFilter_GEMRawToDigi_ME0DigiToRawModule_h
#define EventFilter_GEMRawToDigi_ME0DigiToRawModule_h

/** \class ME0DigiToRawModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"

#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "CondFormats/DataRecord/interface/ME0EMapRcd.h"
#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/GEMObjects/interface/ME0ROmap.h"

namespace edm {
  class ConfigurationDescriptions;
}

class ME0DigiToRawModule : public edm::global::EDProducer<edm::RunCache<ME0ROmap> > {
 public:
  /// Constructor
  ME0DigiToRawModule(const edm::ParameterSet & pset);

  // global::EDProducer
  std::shared_ptr<ME0ROmap> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;  
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  int event_type_;
  edm::EDGetTokenT<ME0DigiCollection> digi_token;
  bool useDBEMap_;  
};
DEFINE_FWK_MODULE(ME0DigiToRawModule);
#endif


