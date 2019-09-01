
#include "Calibration/HcalIsolatedTrackReco/interface/ECALRegFEDSelector.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

ECALRegFEDSelector::ECALRegFEDSelector(const edm::ParameterSet& iConfig) {
  tok_seed_ = consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("regSeedLabel"));
  delta_ = iConfig.getParameter<double>("delta");

  tok_raw_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInputLabel"));

  ec_mapping = std::make_unique<EcalElectronicsMapping>();

  produces<FEDRawDataCollection>();
  produces<EcalListOfFEDS>();

  for (int p = 0; p < 1200; p++) {
    fedSaved[p] = false;
  }
}

ECALRegFEDSelector::~ECALRegFEDSelector() {}

void ECALRegFEDSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  for (int p = 0; p < 1200; p++) {
    fedSaved[p] = false;
  }

  auto producedData = std::make_unique<FEDRawDataCollection>();

  auto fedList = std::make_unique<EcalListOfFEDS>();

  edm::Handle<trigger::TriggerFilterObjectWithRefs> trigSeedTrks;
  iEvent.getByToken(tok_seed_, trigSeedTrks);

  std::vector<edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
  trigSeedTrks->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByToken(tok_raw_, rawIn);

  //  std::vector<int> EC_FED_IDs;

  for (uint32_t p = 0; p < isoPixTrackRefs.size(); p++) {
    double etaObj_ = isoPixTrackRefs[p]->track()->eta();
    double phiObj_ = isoPixTrackRefs[p]->track()->phi();

    RectangularEtaPhiRegion ecEtaPhi(etaObj_ - delta_, etaObj_ + delta_, phiObj_ - delta_, phiObj_ + delta_);

    const std::vector<int> EC_FED_IDs = ec_mapping->GetListofFEDs(ecEtaPhi);

    const FEDRawDataCollection* rdc = rawIn.product();

    for (int j = 0; j < FEDNumbering::MAXFEDID; j++) {
      bool rightFED = false;
      for (uint32_t k = 0; k < EC_FED_IDs.size(); k++) {
        if (j == EcalRegionCabling::fedIndex(EC_FED_IDs[k])) {
          if (!fedSaved[j]) {
            fedList->AddFED(j);
            rightFED = true;
            fedSaved[j] = true;
          }
        }
      }
      if (j >= FEDNumbering::MINPreShowerFEDID && j <= FEDNumbering::MAXPreShowerFEDID) {
        fedSaved[j] = true;
        rightFED = true;
      }
      if (!rightFED)
        continue;
      const FEDRawData& fedData = rdc->FEDData(j);
      size_t size = fedData.size();

      if (size > 0) {
        // this fed has data -- lets copy it
        FEDRawData& fedDataProd = producedData->FEDData(j);
        if (fedDataProd.size() != 0) {
          //		  std::cout << " More than one FEDRawDataCollection with data in FED ";
          //		  std::cout << j << " Skipping the 2nd\n";
          continue;
        }
        fedDataProd.resize(size);
        unsigned char* dataProd = fedDataProd.data();
        const unsigned char* data = fedData.data();
        for (unsigned int k = 0; k < size; ++k) {
          dataProd[k] = data[k];
        }
      }
    }
  }

  iEvent.put(std::move(producedData));
  iEvent.put(std::move(fedList));
}

void ECALRegFEDSelector::beginJob() {}

void ECALRegFEDSelector::endJob() {}
