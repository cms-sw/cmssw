#ifndef EventFilter_GEMRawToDigi_GEMRawToDigiModule_h
#define EventFilter_GEMRawToDigi_GEMRawToDigiModule_h

/** \class GEMRawToDigiModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/GEMObjects/interface/GEMROmap.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "EventFilter/GEMRawToDigi/interface/VFATdata.h"

namespace edm {
   class ConfigurationDescriptions;
}

class GEMRawToDigiModule : public edm::stream::EDProducer<> {
 public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet & pset);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  // Operations
  void produce(edm::Event&, edm::EventSetup const&) override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  
  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
<<<<<<< HEAD
=======
  bool unpackStatusDigis_;
>>>>>>> adding packing and unpacking to std seq
  bool useDBEMap_;
  
  const GEMEMap* m_gemEMap;
  const GEMROmap* m_gemROMap;
  
};
DEFINE_FWK_MODULE(GEMRawToDigiModule);
#endif
