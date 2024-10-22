#ifndef OMTFPatternMaker_H
#define OMTFPatternMaker_H

#include "xercesc/util/XercesDefs.hpp"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

class OMTFProcessor;
class OMTFConfiguration;
class OMTFConfigMaker;
class OMTFinputMaker;

class SimTrack;

class XMLConfigWriter;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}  // namespace XERCES_CPP_NAMESPACE

class OMTFPatternMaker : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  OMTFPatternMaker(const edm::ParameterSet &cfg);

  ~OMTFPatternMaker() override;

  void beginRun(edm::Run const &run, edm::EventSetup const &iSetup) override;

  void endRun(edm::Run const &, edm::EventSetup const &) override;

  void beginJob() override;

  void endJob() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  const SimTrack *findSimMuon(const edm::Event &ev, const SimTrack *previous = nullptr);

  edm::ParameterSet theConfig;
  edm::InputTag g4SimTrackSrc;

  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDTPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDTTh;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  edm::EDGetTokenT<RPCDigiCollection> inputTokenRPC;
  edm::EDGetTokenT<edm::SimTrackContainer> inputTokenSimHit;

  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> esTokenParams_;

  void writeMergedGPs();

  bool makeConnectionsMaps, makeGoldenPatterns, mergeXMLFiles;

  ///Original pdf width. read from configuration.
  unsigned int nPdfAddrBits;

  ///OMTF objects
  OMTFConfiguration *myOMTFConfig;
  OMTFinputMaker *myInputMaker;
  OMTFProcessor *myOMTF;
  ///
  xercesc::DOMElement *aTopElement;
  OMTFConfigMaker *myOMTFConfigMaker;
  XMLConfigWriter *myWriter;
};

#endif
