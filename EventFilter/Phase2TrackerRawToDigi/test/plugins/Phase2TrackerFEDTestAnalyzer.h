#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDTestAnalyzer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDTestAnalyzer_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <utility>
#include <vector>

/**
   @class Phase2TrackerFEDTestAnalyzer 
   @brief Analyzes contents of FED_test_ collection
*/

class Phase2TrackerFEDTestAnalyzer : public edm::EDAnalyzer {
public:
  typedef std::pair<uint16_t, uint16_t> Fed;
  typedef std::vector<Fed> Feds;
  typedef std::vector<uint16_t> Channels;
  typedef std::map<uint16_t, Channels> ChannelsMap;

  Phase2TrackerFEDTestAnalyzer(const edm::ParameterSet&);
  ~Phase2TrackerFEDTestAnalyzer();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  edm::InputTag label_;
  edm::EDGetTokenT<FEDRawDataCollection> token_;
};

#endif  // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDTestAnalyzer_H
