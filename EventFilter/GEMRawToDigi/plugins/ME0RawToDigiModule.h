#ifndef EventFilter_ME0RawToDigiModule_h
#define EventFilter_ME0RawToDigiModule_h

/** \class ME0RawToDigiModule
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
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"

#include "CondFormats/DataRecord/interface/ME0EMapRcd.h"
#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/GEMObjects/interface/ME0ROmap.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "EventFilter/GEMRawToDigi/interface/VFATdata.h"

namespace edm {
   class ConfigurationDescriptions;
}

class ME0RawToDigiModule : public edm::global::EDProducer<> {
 public:
  /// Constructor
  ME0RawToDigiModule(const edm::ParameterSet & pset);

  void doBeginRun_(edm::Run const& rp, edm::EventSetup const& c) override;

  // Operations
  void produce( edm::StreamID, edm::Event&, const edm::EventSetup& ) const override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
  bool useDBEMap_;
  
  const ME0EMap* m_me0EMap;
  ME0ROmap* m_me0ROMap;
  
};
DEFINE_FWK_MODULE(ME0RawToDigiModule);
#endif
