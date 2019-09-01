#ifndef EventFilter_GEMRawToDigi_GEMRawToDigiModule_h
#define EventFilter_GEMRawToDigi_GEMRawToDigiModule_h

/** \class GEMRawToDigiModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMVfatStatusDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMGEBdataCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMAMCdataCollection.h"
#include "EventFilter/GEMRawToDigi/interface/GEMAMC13EventCollection.h"

#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "EventFilter/GEMRawToDigi/interface/VFATdata.h"

namespace edm {
  class ConfigurationDescriptions;
}

class GEMRawToDigiModule : public edm::global::EDProducer<edm::RunCache<GEMROMapping> > {
public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMROMapping> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
  bool useDBEMap_;
  bool unPackStatusDigis_;
};
DEFINE_FWK_MODULE(GEMRawToDigiModule);
#endif
