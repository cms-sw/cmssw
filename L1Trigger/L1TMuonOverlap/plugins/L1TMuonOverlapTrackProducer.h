#ifndef OMTFProducer_H
#define OMTFProducer_H

#include "xercesc/util/XercesDefs.hpp"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"

class L1TMuonOverlapParams;
class OMTFProcessor;
class OMTFConfiguration;
class OMTFConfigMaker;
class XMLConfigWriter;



namespace XERCES_CPP_NAMESPACE{
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}


class L1TMuonOverlapTrackProducer : public edm::EDProducer {
 public:
  L1TMuonOverlapTrackProducer(const edm::ParameterSet&);

  ~L1TMuonOverlapTrackProducer();

  virtual void beginJob();

  virtual void endJob();

  virtual void beginRun(edm::Run const& run, edm::EventSetup const& iSetup);
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::ParameterSet theConfig;
  
  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDTPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDTTh;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  edm::EDGetTokenT<RPCDigiCollection> inputTokenRPC;

  void loadAndFilterDigis(edm::Event& iEvent);

  void getProcessorCandidates(unsigned int iProcessor, l1t::tftype mtfType, int bx,
			      l1t::RegionalMuonCandBxCollection & myCandidates);
  
  void writeResultToXML(unsigned int iProcessor, const OMTFinput &myInput, 
			const std::vector<OMTFProcessor::resultsMap> & myResults);

  bool dumpResultToXML, dumpDetailedResultToXML;

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;

  ///OMTF objects
  OMTFConfiguration *myOMTFConfig;
  OMTFinputMaker myInputMaker;
  OMTFSorter mySorter;
  OMTFProcessor *myOMTF;
  ///
  xercesc::DOMElement *aTopElement;
  OMTFConfigMaker *myOMTFConfigMaker;
  XMLConfigWriter *myWriter;

};

#endif
