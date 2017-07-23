#ifndef EventFilter_GEMRawToDigiModule_h
#define EventFilter_GEMRawToDigiModule_h

/** \class GEMRawToDigiModule
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "EventFilter/GEMRawToDigi/interface/GEMAMC13EventFormat.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

namespace edm {
   class ConfigurationDescriptions;
}

class GEMRawToDigi;

class GEMRawToDigiModule : public edm::EDProducer {
 public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~GEMRawToDigiModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:

  GEMRawToDigi * unPacker;

  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
  AMC13Event * m_AMC13Event;
  
};
#endif


