#ifndef OMTFReconstruction_H
#define OMTFReconstruction_H

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

#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlap/interface/GhostBuster.h"

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

typedef std::vector< std::pair<unsigned int, std::vector<AlgoMuon> > > AlgoMuonResults;

class OMTFReconstruction {
  public:
    OMTFReconstruction();

    OMTFReconstruction(const edm::ParameterSet&);

    ~OMTFReconstruction();

    void beginJob();

    void endJob();

    void beginRun(edm::Run const& run, edm::EventSetup const& iSetup);  

    void algoBeginJob();

    // not working
    AlgoMuonResults algoReconstruct(const edm::Event&, const edm::EventSetup&);

    std::auto_ptr<l1t::RegionalMuonCandBxCollection > reconstruct(const edm::Event&, const edm::EventSetup&);

    // std::auto_ptr<l1t::RegionalMuonCandBxCollection > getAlgoMuonResults();

  // private:

    edm::ParameterSet m_Config;
    bool dumpResultToXML, dumpDetailedResultToXML;

  ///OMTF objects
    OMTFConfiguration   *m_OMTFConfig;
    OMTFinputMaker       m_InputMaker;
    OMTFSorter           m_Sorter;
    OMTFGhostBuster      m_GhostBuster;
    OMTFProcessor       *m_OMTF;
    OMTFProcessor       *m_AlgoOMTF;
    AlgoMuonResults      m_AlgoMuonResults;
  ///
    xercesc::DOMElement *aTopElement;
    OMTFConfigMaker     *m_OMTFConfigMaker;
    XMLConfigWriter     *m_Writer;
	
};

#endif