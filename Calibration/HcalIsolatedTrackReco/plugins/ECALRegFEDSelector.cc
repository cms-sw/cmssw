// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

class ECALRegFEDSelector : public edm::one::EDProducer<> {
public:
  ECALRegFEDSelector(const edm::ParameterSet&);
  ~ECALRegFEDSelector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  std::unique_ptr<const EcalElectronicsMapping> ec_mapping;

  const double delta_;
  bool fedSaved[1200];

  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_seed_;
  const edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
};

ECALRegFEDSelector::ECALRegFEDSelector(const edm::ParameterSet& iConfig)
    : delta_(iConfig.getParameter<double>("delta")),
      tok_seed_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("regSeedLabel"))),
      tok_raw_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("rawInputLabel"))) {
  ec_mapping = std::make_unique<EcalElectronicsMapping>();

  produces<FEDRawDataCollection>();
  produces<EcalListOfFEDS>();

  for (int p = 0; p < 1200; p++) {
    fedSaved[p] = false;
  }
}

void ECALRegFEDSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  for (int p = 0; p < 1200; p++) {
    fedSaved[p] = false;
  }

  auto producedData = std::make_unique<FEDRawDataCollection>();

  auto fedList = std::make_unique<EcalListOfFEDS>();

  const edm::Handle<trigger::TriggerFilterObjectWithRefs>& trigSeedTrks = iEvent.getHandle(tok_seed_);

  std::vector<edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
  trigSeedTrks->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

  const edm::Handle<FEDRawDataCollection>& rawIn = iEvent.getHandle(tok_raw_);

  //  std::vector<int> EC_FED_IDs;

  for (uint32_t p = 0; p < isoPixTrackRefs.size(); p++) {
    double etaObj_ = isoPixTrackRefs[p]->track()->eta();
    double phiObj_ = isoPixTrackRefs[p]->track()->phi();

    RectangularEtaPhiRegion ecEtaPhi(etaObj_ - delta_, etaObj_ + delta_, phiObj_ - delta_, phiObj_ + delta_);

    const std::vector<int> EC_FED_IDs = ec_mapping->GetListofFEDs(ecEtaPhi);

    const FEDRawDataCollection* rdc = rawIn.product();

    for (int j = 0; j <= FEDNumbering::MAXFEDID; j++) {
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
          edm::LogVerbatim("HcalCalib") << " More than one FEDRawDataCollection with data in FED " << j
                                        << " Skipping the 2nd";
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

void ECALRegFEDSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("regSeedLabel", edm::InputTag("hltPixelIsolTrackFilter"));
  desc.add<edm::InputTag>("rawInputLabel", edm::InputTag("rawDataCollector"));
  desc.add<double>("delta", 1.0);
  descriptions.add("ecalFED", desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ECALRegFEDSelector);
