#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"

#include "EventFilter/L1ScoutingRawToDigi/interface/shifts.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/scales.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"

class ScCaloRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScCaloRawToDigi(const edm::ParameterSet&);
  ~ScCaloRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void unpackOrbit(
    //l1t::JetBxCollection* jets, l1t::TauBxCollection* taus,
    //l1t::EGammaBxCollection* eGammas, l1t::EtSumBxCollection* etSums,
    scoutingRun3::ScJetOrbitCollection* jets, scoutingRun3::ScTauOrbitCollection* taus,
    scoutingRun3::ScEGammaOrbitCollection* eGammas, scoutingRun3::ScEtSumOrbitCollection* etSums,
    const unsigned char* buf, size_t len
  );

  void unpackRawJet(std::vector<l1t::Jet>& jets, uint32_t *rawData);

  // std::vector<l1t::Jet> bx_jets;
  // std::vector<l1t::Tau> bx_taus;
  // std::vector<l1t::EGamma> bx_eGammas;
  // std::vector<l1t::EtSum> bx_etSums;

  std::unique_ptr<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>> dummyLVec_;

  bool debug = false;

  edm::InputTag srcInputTag;
  edm::EDGetToken rawToken;
};
