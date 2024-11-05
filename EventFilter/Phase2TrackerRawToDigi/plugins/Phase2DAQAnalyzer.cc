// Phase2DAQAnalyzer.cc
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "TH1F.h"
#include "TFile.h"
#include <fstream>
#include <iostream>
#include <bitset>

class Phase2DAQAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit Phase2DAQAnalyzer(const edm::ParameterSet&);
  ~Phase2DAQAnalyzer() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  edm::EDGetTokenT<FEDRawDataCollection> fedRawDataToken_;
};

Phase2DAQAnalyzer::Phase2DAQAnalyzer(const edm::ParameterSet& iConfig) :
    fedRawDataToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection")))
{ }

void Phase2DAQAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<FEDRawDataCollection> fedRawDataCollection;
  iEvent.getByToken(fedRawDataToken_, fedRawDataCollection);

  if (!fedRawDataCollection.isValid()) 
  {
    edm::LogError("Phase2DAQAnalyzer") << "No FEDRawDataCollection found!";
    return;
  }

  // Read only the 0th FED position as per the producer logic
  const FEDRawData& fedData = fedRawDataCollection->FEDData(0);  // FED ID 0
 

  // ** Below is the logic to read out the 32bit words from the fedRawData object.

    if (fedData.size() > 0) 
    {
      const unsigned char* dataPtr = fedData.data();

      for (size_t i = 0; i < fedData.size(); i += 4)  // Read 4 bytes (32 bits) at a time
      {
          // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          uint32_t word = (static_cast<uint32_t>(dataPtr[i]) << 24) | 
                          (static_cast<uint32_t>(dataPtr[i + 1]) << 16) | 
                          (static_cast<uint32_t>(dataPtr[i + 2]) << 8) | 
                          (static_cast<uint32_t>(dataPtr[i + 3]));

          // // Print hexadecimal representation
          std::cout << std::hex << std::setw(8) << std::setfill('0') 
                    << word << "  ";

          // Print binary representation
          std::cout << std::bitset<32>(word) << std::endl;
      }
    }
}

void Phase2DAQAnalyzer::endJob() 
{

}

// Define this as a plug-in
DEFINE_FWK_MODULE(Phase2DAQAnalyzer);
