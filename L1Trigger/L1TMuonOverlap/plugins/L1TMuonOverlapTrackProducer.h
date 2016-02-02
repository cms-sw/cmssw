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

#include "TRandom3.h"

class L1TMuonOverlapParams;
class OMTFProcessor;
class OMTFConfiguration;
class OMTFConfigMaker;
class OMTFinputMaker;
class OMTFSorter;
class OMTFinput;
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

  void processCandidates(unsigned int iProcessor, int bx,
			 std::auto_ptr<l1t::RegionalMuonCandBxCollection > & myCands,
			 l1t::RegionalMuonCandBxCollection & myOTFCandidates,
			 l1t::tftype mtfType);

  void writeMergedGPs();

  bool dumpResultToXML, dumpDetailedResultToXML, dumpGPToXML;

  ///OMTF objects
  OMTFConfiguration *myOMTFConfig;
  OMTFinputMaker *myInputMaker;
  OMTFSorter *mySorter;
  OMTFProcessor *myOMTF;
  OMTFinput *myInputXML;
  ///
  xercesc::DOMElement *aTopElement;
  OMTFConfigMaker *myOMTFConfigMaker;
  XMLConfigWriter *myWriter;
  std::shared_ptr<L1TMuonOverlapParams> omtfParams;
  ///

};

#endif
