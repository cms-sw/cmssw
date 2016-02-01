
#ifndef OMTFProducerMix_H
#define OMTFProducerMix_H

#include "xercesc/util/XercesDefs.hpp"

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
class XMLConfigReader;

namespace XERCES_CPP_NAMESPACE{
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}


class OMTFProducerMix : public edm::EDProducer {
 public:
  OMTFProducerMix(const edm::ParameterSet&);
  
  ~OMTFProducerMix();

  virtual void beginRun(edm::Run const& run, edm::EventSetup const& iSetup);

  virtual void beginJob();

  virtual void endJob();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);  

 private:

  edm::ParameterSet theConfig;

  edm::EDGetTokenT<L1MuDTChambPhContainer> inputTokenDTPh;
  edm::EDGetTokenT<L1MuDTChambThContainer> inputTokenDTTh;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> inputTokenCSC;
  edm::EDGetTokenT<RPCDigiCollection> inputTokenRPC;

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
  XMLConfigReader *myReader;
  ///
  unsigned int myEventNumber;
  unsigned int eventsToMix;
  bool dumpResultToXML;
  TRandom3 aRndm;


};

#endif
