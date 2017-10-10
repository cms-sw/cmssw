#ifndef EventFilter_GEMDigiToRawModule_h
#define EventFilter_GEMDigiToRawModule_h

/** \class GEMDigiToRawModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

class GEMDigiToRaw;

class GEMDigiToRawModule : public edm::EDProducer {
 public:
  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;
  
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~GEMDigiToRawModule(){}

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  // ------------ method called once each 64 Bits data word and keep in vector ------------
  void ByteVector(std::vector<unsigned char>&, uint64_t&);
  
  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection>             digi_token;
  
};
#endif


