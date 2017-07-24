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
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

namespace edm {
   class ConfigurationDescriptions;
}

class GEMDigiToRaw;

class GEMDigiToRawModule : public edm::EDProducer {
 public:
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~GEMDigiToRawModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection>             digi_token;
  edm::EDGetTokenT<GEMPadDigiCollection>          padDigi_token;
  edm::EDGetTokenT<GEMPadDigiClusterCollection>   padDigiCluster_token;
  edm::EDGetTokenT<GEMCoPadDigiCollection>        coPadDigi_token;
  
};
#endif


