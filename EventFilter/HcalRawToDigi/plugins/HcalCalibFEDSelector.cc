// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalFEDList.h"

class HcalCalibFEDSelector : public edm::global::EDProducer<> {
public:
  HcalCalibFEDSelector(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<FEDRawDataCollection> tok_fed_;
  std::vector<int> extraFEDs_;
};

HcalCalibFEDSelector::HcalCalibFEDSelector(const edm::ParameterSet& iConfig) {
  tok_fed_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInputLabel"));
  extraFEDs_ = iConfig.getParameter<std::vector<int> >("extraFEDsToKeep");
  produces<FEDRawDataCollection>();
}

void HcalCalibFEDSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto producedData = std::make_unique<FEDRawDataCollection>();

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByToken(tok_fed_, rawIn);

  std::vector<int> selFEDs;

  //--- Get the list of FEDs to be kept ---//
  int calibType = -1;
  for (int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
    const FEDRawData& fedData = rawIn->FEDData(i);
    if (fedData.size() < 24)
      continue;  // FED is empty
    int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType();
    if (calibType < 0) {
      calibType = value;
    } else {
      if (calibType != value)
        edm::LogWarning("HcalCalibFEDSelector") << "Conflicting calibration types found: " << calibType << " vs. "
                                                << value << ".  Staying with " << calibType;
    }
  }

  HcalFEDList calibFeds(calibType);
  selFEDs = calibFeds.getListOfFEDs();
  for (unsigned int i = 0; i < extraFEDs_.size(); i++) {
    bool duplicate = false;
    for (unsigned int j = 0; j < selFEDs.size(); j++) {
      if (extraFEDs_.at(i) == selFEDs.at(j)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
      selFEDs.push_back(extraFEDs_.at(i));
  }

  // Copying:
  const FEDRawDataCollection* rdc = rawIn.product();

  for (int j = 0; j < FEDNumbering::lastFEDId(); ++j) {
    bool rightFED = false;
    for (uint32_t k = 0; k < selFEDs.size(); k++) {
      if (j == selFEDs[k]) {
        rightFED = true;
      }
    }
    if (!rightFED)
      continue;
    const FEDRawData& fedData = rdc->FEDData(j);
    size_t size = fedData.size();

    if (size > 0) {
      // this fed has data -- lets copy it
      FEDRawData& fedDataProd = producedData->FEDData(j);
      if (fedDataProd.size() != 0) {
        continue;
      }
      fedDataProd.resize(size);
      unsigned char* dataProd = fedDataProd.data();
      const unsigned char* data = fedData.data();
      // memcpy is at-least-as-fast as assignment and can be much faster
      memcpy(dataProd, data, size);
    }
  }

  iEvent.put(std::move(producedData));
}

DEFINE_FWK_MODULE(HcalCalibFEDSelector);
