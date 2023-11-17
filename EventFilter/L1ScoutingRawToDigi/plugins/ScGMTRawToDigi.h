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

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"

#include "EventFilter/L1ScoutingRawToDigi/interface/shifts.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/scales.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"

class ScGMTRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScGMTRawToDigi(const edm::ParameterSet&);
  ~ScGMTRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  //void endStream() override;

  void unpackOrbit(
    scoutingRun3::ScMuonOrbitCollection* muons,
    //l1t::MuonBxCollection* muons,
    const unsigned char* buf, size_t len
  );

  std::vector<l1t::Muon> bx_muons;
  std::unique_ptr<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>> dummyLVec_;

  bool debug_ = false;

  edm::InputTag srcInputTag;
  edm::EDGetToken rawToken;
};
