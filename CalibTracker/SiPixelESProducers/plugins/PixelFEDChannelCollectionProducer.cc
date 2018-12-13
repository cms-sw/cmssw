// -*- C++ -*-
//
// Package:    CalibTracker/PixelFEDChannelCollectionProducer
// Class:      PixelFEDChannelCollectionProducer
// 
/**\class PixelFEDChannelCollectionProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 13 Dec 2018 08:48:22 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h" 
#include "CondFormats/SiPixelObjects/interface/SiPixelFEDChannelContainer.h"
#include "CalibTracker/Records/interface/SiPixelFEDChannelContainerESProducerRcd.h"

// Need to add #include statements for definitions of
// the data type and record type here

//
// class declaration
//

class PixelFEDChannelCollectionProducer : public edm::ESProducer {
   public:
      PixelFEDChannelCollectionProducer(const edm::ParameterSet&);
      ~PixelFEDChannelCollectionProducer();

      typedef std::unordered_map<std::string,PixelFEDChannelCollection> PixelFEDChannelCollectionMap;

      using ReturnType = std::unique_ptr<PixelFEDChannelCollectionMap>;

      ReturnType produce(const SiPixelFEDChannelContainerESProducerRcd &);
   private:
      // ----------member data ---------------------------
};


PixelFEDChannelCollectionProducer::PixelFEDChannelCollectionProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


PixelFEDChannelCollectionProducer::~PixelFEDChannelCollectionProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to produce the data  ------------
PixelFEDChannelCollectionProducer::ReturnType
PixelFEDChannelCollectionProducer::produce(const SiPixelFEDChannelContainerESProducerRcd& iRecord)
{
  edm::ESHandle<SiPixelFEDChannelContainer> qualityCollectionHandle;
  iRecord.getRecord<SiPixelStatusScenariosRcd>().get(qualityCollectionHandle); 

  PixelFEDChannelCollectionMap out;

  for(const auto& it : qualityCollectionHandle->getScenarioMap()){
 
    std::string scenario = it.first;
    PixelFEDChannelCollection disabled_channelcollection;
    auto SiPixelBadFedChannels = it.second;
    for(const auto &entry : SiPixelBadFedChannels){
      disabled_channelcollection.insert(entry.first, entry.second.data(), entry.second.size());
    }
    out.emplace(scenario,disabled_channelcollection);
  }
  
  auto product = std::make_unique<PixelFEDChannelCollectionMap>(out);
  return product;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(PixelFEDChannelCollectionProducer);
