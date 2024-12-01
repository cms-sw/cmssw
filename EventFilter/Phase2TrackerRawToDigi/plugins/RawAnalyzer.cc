#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include <iostream>
#include <bitset>

class RawAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit RawAnalyzer(const edm::ParameterSet&);
  ~RawAnalyzer() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  edm::EDGetTokenT<FEDRawDataCollection> fedRawDataToken_;
};

RawAnalyzer::RawAnalyzer(const edm::ParameterSet& iConfig) :
    fedRawDataToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection")))
{ }

void RawAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<FEDRawDataCollection> fedRawDataCollection;
  iEvent.getByToken(fedRawDataToken_, fedRawDataCollection);

  if (!fedRawDataCollection.isValid()) 
  {
    edm::LogError("RawAnalyzer") << "No FEDRawDataCollection found!";
    return;
  }

  // Read only the 0th FED position as per the producer logic
  const FEDRawData& fedData_slink_0 = fedRawDataCollection->FEDData(0 + 4 * (1 - 1) + 0);  // FED ID 0
  const FEDRawData& fedData_slink_1 = fedRawDataCollection->FEDData(1 + 4 * (1 - 1) + 0);  // FED ID 1
  const FEDRawData& fedData_slink_2 = fedRawDataCollection->FEDData(2 + 4 * (1 - 1) + 0);  // FED ID 2
  const FEDRawData& fedData_slink_3 = fedRawDataCollection->FEDData(3 + 4 * (1 - 1) + 0);  // FED ID 3
 

  // ** Below is the logic to read out the 32bit words from the fedRawData object.

  std::vector<const FEDRawData*> fedDataVec = {&fedData_slink_0, &fedData_slink_1, &fedData_slink_2, &fedData_slink_3};

  // Determine the maximum size among all FEDRawData objects
  size_t maxWords = 0;
  for (const auto& fedData : fedDataVec) {
      maxWords = std::max(maxWords, fedData->size() / 4);  // Divide by 4 to get 32-bit words
  }

  // Prepare column headers
  std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------- SLink 0 -------------------------------- SLink 1 -------------------------------- SLink 2 -------------------------------- SLink 3 ----------------" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

  // Loop through all rows (up to maxWords) and print each 32-bit word
  for (size_t row = 0; row < maxWords; ++row) 
  {
      for (const auto& fedData : fedDataVec) 
      {
          if (row * 4 < fedData->size()) 
          {
              const unsigned char* dataPtr = fedData->data();
              uint32_t word = (static_cast<uint32_t>(dataPtr[row * 4]) << 24) |
                              (static_cast<uint32_t>(dataPtr[row * 4 + 1]) << 16) |
                              (static_cast<uint32_t>(dataPtr[row * 4 + 2]) << 8) |
                              (static_cast<uint32_t>(dataPtr[row * 4 + 3]));

              // Print hexadecimal and binary representations
              std::cout << std::hex << std::setw(8) << std::setfill('0') << word << " "
                        << std::bitset<32>(word) << " ";
          } else {
              // Print empty space for missing data in this column
              std::cout << "                  ";
          }
      }
      std::cout << std::endl;  // Move to the next line after printing a row
  }

}

void RawAnalyzer::endJob() 
{

}

// Define this as a plug-in
DEFINE_FWK_MODULE(RawAnalyzer);
