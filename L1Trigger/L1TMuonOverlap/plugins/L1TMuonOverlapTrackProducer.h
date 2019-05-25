#ifndef OMTFProducer_H
#define OMTFProducer_H

#include "xercesc/util/XercesDefs.hpp"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFReconstruction.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"

class L1TMuonOverlapParams;
class OMTFProcessor;
class OMTFConfiguration;
class OMTFConfigMaker;
class XMLConfigWriter;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}  // namespace XERCES_CPP_NAMESPACE

class L1TMuonOverlapTrackProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
public:
  L1TMuonOverlapTrackProducer(const edm::ParameterSet&);

  ~L1TMuonOverlapTrackProducer() override;

  void beginJob() override;

  void endJob() override;

  void beginRun(edm::Run const& run, edm::EventSetup const& iSetup) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::ParameterSet theConfig;

  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDTPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDTTh;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  edm::EDGetTokenT<RPCDigiCollection> inputTokenRPC;

  bool dumpResultToXML, dumpDetailedResultToXML;

  OMTFReconstruction m_Reconstruction;
};

#endif
