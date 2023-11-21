#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"

#include "EventFilter/L1ScoutingRawToDigi/interface/shifts.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/scales.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/conversion.h"

class ScCaloRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScCaloRawToDigi(const edm::ParameterSet&);
  ~ScCaloRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void unpackOrbit(
    //scoutingRun3::ScJetOrbitCollection* jets, scoutingRun3::ScTauOrbitCollection* taus,
    //scoutingRun3::ScEGammaOrbitCollection* eGammas, scoutingRun3::ScEtSumOrbitCollection* etSums,
    const unsigned char* buf, size_t len
  );

  // void unpackLinkJets(scoutingRun3::ScJetOrbitCollection* jets, uint32_t* dataBlock, int bx);
  // void unpackLinkEGammas(scoutingRun3::ScEGammaOrbitCollection* eGammas, uint32_t* dataBlock, int bx);
  // void unpackLinkTaus(scoutingRun3::ScTauOrbitCollection* taus, uint32_t* dataBlock, int bx);
  // void unpackEtSums(scoutingRun3::ScEtSumOrbitCollection* etSums, uint32_t* dataBlock, int bx);

  void unpackLinkJets(uint32_t* dataBlock, int bx);
  void unpackLinkEGammas(uint32_t* dataBlock, int bx);
  void unpackLinkTaus(uint32_t* dataBlock, int bx);
  void unpackEtSums(uint32_t* dataBlock, int bx);

  int nJetsOrbit_, nEGammasOrbit_, nTausOrbit_, nEtSumsOrbit_;
  std::vector<std::vector<scoutingRun3::ScJet>> orbitBufferJets_;
  std::vector<std::vector<scoutingRun3::ScEGamma>> orbitBufferEGammas_;
  std::vector<std::vector<scoutingRun3::ScTau>> orbitBufferTaus_;
  std::vector<std::vector<scoutingRun3::ScEtSum>> orbitBufferEtSums_;

  bool debug_ = false;
  edm::InputTag srcInputTag;
  edm::EDGetToken rawToken;
};
