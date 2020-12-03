#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "CalibTracker/Records/interface/SiPixelFEDChannelContainerESProducerRcd.h"

class PixelFEDChannelCollectionMapTestReader : public edm::one::EDAnalyzer<> {
public:
  typedef std::unordered_map<std::string, PixelFEDChannelCollection> PixelFEDChannelCollectionMap;
  explicit PixelFEDChannelCollectionMapTestReader(edm::ParameterSet const& p);
  ~PixelFEDChannelCollectionMapTestReader();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // ----------member data ---------------------------
  const bool printdebug_;
  const std::string formatedOutput_;
  edm::ESGetToken<PixelFEDChannelCollectionMap, SiPixelFEDChannelContainerESProducerRcd>
      pixelFEDChannelCollectionMapToken_;
};

PixelFEDChannelCollectionMapTestReader::PixelFEDChannelCollectionMapTestReader(edm::ParameterSet const& p)
    : printdebug_(p.getUntrackedParameter<bool>("printDebug", true)),
      formatedOutput_(p.getUntrackedParameter<std::string>("outputFile", "")),
      pixelFEDChannelCollectionMapToken_(
          esConsumes<PixelFEDChannelCollectionMap, SiPixelFEDChannelContainerESProducerRcd>()) {
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "PixelFEDChannelCollectionMapTestReader" << std::endl;
}

PixelFEDChannelCollectionMapTestReader::~PixelFEDChannelCollectionMapTestReader() {
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "~PixelFEDChannelCollectionMapTestReader " << std::endl;
}

void PixelFEDChannelCollectionMapTestReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader")
      << "### PixelFEDChannelCollectionMapTestReader::analyze  ###" << std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << " ---EVENT NUMBER " << e.id().event() << std::endl;

  edm::eventsetup::EventSetupRecordKey recordKey(
      edm::eventsetup::EventSetupRecordKey::TypeTag::findType("'SiPixelFEDChannelContainerESProducerRcd"));

  if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    //record not found
    edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "Record \"SiPixelFEDChannelContainerESProducerRcd"
                                                           << "\" does not exist " << std::endl;
  }

  //this part gets the handle of the event source and the record (i.e. the Database)
  edm::ESHandle<PixelFEDChannelCollectionMap> PixelFEDChannelCollectionMapHandle =
      context.getHandle(pixelFEDChannelCollectionMapToken_);
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "got eshandle and context" << std::endl;

  const PixelFEDChannelCollectionMap* thePixelFEDChannelCollectionMap = PixelFEDChannelCollectionMapHandle.product();
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "got SiPixelFEDChannelContainer* " << std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader")
      << "print  pointer address : " << thePixelFEDChannelCollectionMap << std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader")
      << "Size: " << thePixelFEDChannelCollectionMap->size() << std::endl;
  edm::LogInfo("PixelFEDChannelCollectionMapTestReader") << "Content of my PixelFEDChanneCollectionlMap: " << std::endl;

  FILE* pFile = NULL;
  if (formatedOutput_ != "")
    pFile = fopen(formatedOutput_.c_str(), "w");
  if (pFile) {
    fprintf(pFile, "PixelFEDChannelCollectionMap::printAll() \n");
    fprintf(pFile,
            " ========================================================================================================="
            "========== \n");

    for (auto it = thePixelFEDChannelCollectionMap->begin(); it != thePixelFEDChannelCollectionMap->end(); ++it) {
      fprintf(pFile,
              " ======================================================================================================="
              "============ \n");
      fprintf(pFile, "run : %s \n ", (it->first).c_str());
      const auto& thePixelFEDChannels = it->second;

      //std:: cout << thePixelFEDChannels.size() << std::endl;
      // for (edmNew::DetSetVector<PixelFEDChannel>::const_iterator DSViter = thePixelFEDChannels.begin(); DSViter != thePixelFEDChannels.end(); DSViter++) {
      //       unsigned int theDetId = DSViter->id();
      //       std:: cout << theDetId << std::endl;
      // }

      for (const auto& disabledChannels : thePixelFEDChannels) {
        //loop over different PixelFED in a PixelFED vector (module)
        fprintf(pFile, "DetId : %i \n", disabledChannels.detId());
        for (const auto& ch : disabledChannels) {
          fprintf(pFile,
                  "fed : %i | link : %2i | roc_first : %2i | roc_last: %2i \n",
                  ch.fed,
                  ch.link,
                  ch.roc_first,
                  ch.roc_last);
          //std::cout <<  disabledChannels.detId() << " "<< ch.fed << " " << ch.link << " " << ch.roc_first << " " << ch.roc_last << std::endl;
        }  // loop over disable channels
      }    // loop over the detSetVector
    }      // main loop on the map
  }        // if file exists
}

void PixelFEDChannelCollectionMapTestReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Reads payloads of type SiPixelFEDChannelContainer");
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<std::string>("outputFile", "");
  descriptions.add("PixelFEDChannelCollectionMapTestReader", desc);
}

DEFINE_FWK_MODULE(PixelFEDChannelCollectionMapTestReader);
