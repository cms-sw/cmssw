#ifndef DQM_L1TMonitor_L1TStage2RegionalShower_h
#define DQM_L1TMonitor_L1TStage2RegionalShower_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

class L1TStage2RegionalShower : public DQMOneEDAnalyzer<> {
public:
  L1TStage2RegionalShower(const edm::ParameterSet& ps);
  ~L1TStage2RegionalShower() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<l1t::RegionalMuonShowerBxCollection> EMTFShowerToken;
  edm::EDGetTokenT<CSCShowerDigiCollection> CSCShowerToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* cscShowerOccupancyLoose;
  MonitorElement* cscShowerOccupancyNom;
  MonitorElement* cscShowerOccupancyTight;
  MonitorElement* cscShowerStationRing;
  MonitorElement* cscShowerChamber;

  MonitorElement* emtfShowerTypeOccupancy;
};

#endif
