#ifndef EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include <utility>
#include <vector>

/**
   @class SiStripFEDRawDataAnalyzer 
   @brief Analyzes contents of FEDRawData collection
*/

class SiStripFEDRawDataAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef std::pair<uint16_t, uint16_t> Fed;
  typedef std::vector<Fed> Feds;
  typedef std::vector<uint16_t> Channels;
  typedef std::map<uint16_t, Channels> ChannelsMap;

  SiStripFEDRawDataAnalyzer(const edm::ParameterSet&);
  ~SiStripFEDRawDataAnalyzer();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

private:
  const edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> esTokenCabling_;
  edm::InputTag label_;
};

#endif  // EventFilter_SiStripRawToDigi_SiStripFEDRawDataAnalyzer_H
