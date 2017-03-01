
#ifndef EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H
#define EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "boost/cstdint.hpp"
#include <string>
namespace edm {
  class ConfigurationDescriptions;
}

namespace sistrip {

  class DigiToRaw;

  /**
     @file EventFilter/SiStripRawToDigi/interface/SiStripDigiToRawModule.h
     @class DigiToRawModule 
   
     @brief A plug-in module that takes StripDigis as input from the
     Event and creates an EDProduct comprising a FEDRawDataCollection.
  */
  class dso_hidden DigiToRawModule final : public edm::stream::EDProducer<> {
  
  public:
  
    DigiToRawModule( const edm::ParameterSet& );
    ~DigiToRawModule();
  
    virtual void beginJob() {}
    virtual void endJob() {}
  
    virtual void produce( edm::Event&, const edm::EventSetup& );
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
  private:

    std::string inputModuleLabel_;
    std::string inputDigiLabel_;
    //CAMM can we do without this bool based on the mode ?
    bool copyBufferHeader_;
    FEDReadoutMode mode_;
    bool rawdigi_;
    DigiToRaw* digiToRaw_;
    uint32_t eventCounter_;
    edm::EDGetTokenT< edm::DetSetVector<SiStripRawDigi> > tokenRawDigi;
    edm::EDGetTokenT< edm::DetSetVector<SiStripDigi> > tokenDigi;
    edm::InputTag rawDataTag_;
    edm::EDGetTokenT<FEDRawDataCollection> tokenRawBuffer;

  };

}

#endif // EventFilter_SiStripRawToDigi_SiStripDigiToRawModule_H

