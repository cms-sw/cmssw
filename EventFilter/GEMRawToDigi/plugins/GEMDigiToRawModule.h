#ifndef EventFilter_GEMDigiToRawModule_h
#define EventFilter_GEMDigiToRawModule_h

/** \class GEMDigiToRawModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/EDProducer.h>
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

class GEMDigiToRaw;

class GEMDigiToRawModule : public edm::EDProducer {
 public:
  
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~GEMDigiToRawModule(){}

  virtual void beginRun(const edm::Run &, const edm::EventSetup&);

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  uint16_t checkCRC(uint8_t b1010, uint16_t BC, uint8_t b1100,
		    uint8_t EC, uint8_t Flag, uint8_t b1110,
		    uint16_t ChipID, uint64_t msData, uint64_t lsData);
  uint16_t crc_cal(uint16_t crc_in, uint16_t dato);
  
  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection> digi_token;
  bool useDBEMap_;

  const GEMEMap* m_gemEMap;
  GEMROmap* m_gemROMap;
  
};
DEFINE_FWK_MODULE(GEMDigiToRawModule);
#endif


