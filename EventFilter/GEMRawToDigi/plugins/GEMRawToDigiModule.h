#ifndef EventFilter_GEMRawToDigiModule_h
#define EventFilter_GEMRawToDigiModule_h

/** \class GEMRawToDigiModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/EDProducer.h>
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

class GEMRawToDigiModule : public edm::EDProducer {
 public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~GEMRawToDigiModule(){}

  virtual void beginRun(const edm::Run &, const edm::EventSetup&);

  // Operations
  virtual void produce(edm::Event&, const edm::EventSetup&);

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  uint16_t checkCRC(gem::VFATdata * m_vfatdata);
  uint16_t crc_cal(uint16_t crc_in, uint16_t dato);
  
  edm::EDGetTokenT<FEDRawDataCollection> fed_token;

  const GEMEMap* m_gemEMap;
  GEMROmap* m_gemROMap;
  
};
DEFINE_FWK_MODULE(GEMRawToDigiModule);
#endif
