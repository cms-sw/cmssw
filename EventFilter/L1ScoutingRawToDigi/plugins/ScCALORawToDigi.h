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
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"
#include "L1TriggerScouting/Utilities/interface/printScObjects.h"

#include <iostream>
#include <memory>

class ScCaloRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScCaloRawToDigi(const edm::ParameterSet&);
  ~ScCaloRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void unpackOrbit(const unsigned char* buf, size_t len);

  void unpackLinkJets(uint32_t* dataBlock, int bx);
  void unpackLinkEGammas(uint32_t* dataBlock, int bx);
  void unpackLinkTaus(uint32_t* dataBlock, int bx);
  void unpackEtSums(uint32_t* dataBlock, int bx);

  int nJetsOrbit_, nEGammasOrbit_, nTausOrbit_, nEtSumsOrbit_;
  // vectors holding data for every bunch crossing
  // before  filling the orbit collection
  std::vector<std::vector<l1ScoutingRun3::Jet>> orbitBufferJets_;
  std::vector<std::vector<l1ScoutingRun3::EGamma>> orbitBufferEGammas_;
  std::vector<std::vector<l1ScoutingRun3::Tau>> orbitBufferTaus_;
  std::vector<std::vector<l1ScoutingRun3::BxSums>> orbitBufferEtSums_;

  bool debug_ = false;
  bool enableAllSums_ = false;
  edm::InputTag srcInputTag;
  edm::EDGetToken rawToken;
};
