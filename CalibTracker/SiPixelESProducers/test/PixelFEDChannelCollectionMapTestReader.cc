#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "CalibTracker/Records/interface/SiPixelFEDChannelContainerESProducerRcd.h"

class PixelFEDChannelCollectionMapTestReader : public edm::one::EDAnalyzer<> {
public:

  typedef std::unordered_map<std::string,PixelFEDChannelCollection> PixelFEDChannelCollectionMap;
  explicit PixelFEDChannelCollectionMapTestReader(edm::ParameterSet const& p); 
  ~PixelFEDChannelCollectionMapTestReader(); 

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
    
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------
  const bool printdebug_;
  const std::string formatedOutput_;
  
};
  
PixelFEDChannelCollectionMapTestReader::PixelFEDChannelCollectionMapTestReader(edm::ParameterSet const& p):
  printdebug_(p.getUntrackedParameter<bool>("printDebug",true)),
  formatedOutput_(p.getUntrackedParameter<std::string>("outputFile",""))
{ 
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader")<<"PixelFEDChannelCollectionMapTestReader"<<std::endl;
}

PixelFEDChannelCollectionMapTestReader::~PixelFEDChannelCollectionMapTestReader() {  
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader")<<"~PixelFEDChannelCollectionMapTestReader "<<std::endl;
}

void
PixelFEDChannelCollectionMapTestReader::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"### PixelFEDChannelCollectionMapTestReader::analyze  ###"<<std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("'SiPixelFEDChannelContainerESProducerRcd"));
    
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"Record \"SiPixelFEDChannelContainerESProducerRcd"<<"\" does not exist "<<std::endl;
  }
  
  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<PixelFEDChannelCollectionMap> PixelFEDChannelCollectionMapHandle;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"got eshandle"<<std::endl;
  
  context.get<SiPixelFEDChannelContainerESProducerRcd>().get(PixelFEDChannelCollectionMapHandle);
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"got context"<<std::endl;

  const PixelFEDChannelCollectionMap* quality_map=PixelFEDChannelCollectionMapHandle.product();
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"got SiPixelFEDChannelContainer* "<< std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "print  pointer address : " ;
  //edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << quality_map << std::endl;
  
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "Size "  <<  quality_map->size() << std::endl;     
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") <<"Content of myQuality_Map "<<std::endl;

}

void
PixelFEDChannelCollectionMapTestReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Reads payloads of type SiPixelFEDChannelContainer");
  desc.addUntracked<bool>("printDebug",true);
  desc.addUntracked<std::string>("outputFile","");
  descriptions.add("PixelFEDChannelCollectionMapTestReader",desc);
}

DEFINE_FWK_MODULE(PixelFEDChannelCollectionMapTestReader);
