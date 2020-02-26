#ifndef OMTFReconstruction_H
#define OMTFReconstruction_H

#include "xercesc/util/XercesDefs.hpp"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlap/interface/GhostBuster.h"

class OMTFProcessor;
class OMTFConfiguration;
class OMTFConfigMaker;
class XMLConfigWriter;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMDocument;
  class DOMImplementation;
}  // namespace XERCES_CPP_NAMESPACE

class OMTFReconstruction {
public:
  OMTFReconstruction(const edm::ParameterSet &, edm::ConsumesCollector &&);

  ~OMTFReconstruction();

  void beginJob();

  void endJob();

  void beginRun(edm::Run const &, edm::EventSetup const &);

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> reconstruct(const edm::Event &, const edm::EventSetup &);

private:
  edm::ParameterSet m_Config;

  edm::Handle<L1MuDTChambPhContainer> dtPhDigis;
  edm::Handle<L1MuDTChambThContainer> dtThDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> cscDigis;
  edm::Handle<RPCDigiCollection> rpcDigis;

  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> l1TMuonOverlapParamsToken_;

  void loadAndFilterDigis(const edm::Event &);

  void getProcessorCandidates(unsigned int iProcessor,
                              l1t::tftype mtfType,
                              int bx,
                              l1t::RegionalMuonCandBxCollection &myCandidates);

  void writeResultToXML(unsigned int iProcessor,
                        l1t::tftype mtfType,
                        const OMTFinput &myInput,
                        const std::vector<OMTFProcessor::resultsMap> &myResults,
                        const std::vector<l1t::RegionalMuonCand> &candMuons);

  bool dumpResultToXML, dumpDetailedResultToXML;
  int bxMin, bxMax;

  ///OMTF objects
  OMTFConfiguration *m_OMTFConfig;
  OMTFinputMaker m_InputMaker;
  OMTFSorter m_Sorter;
  std::unique_ptr<IGhostBuster> m_GhostBuster;
  OMTFProcessor *m_OMTF;
  ///
  xercesc::DOMElement *aTopElement;
  OMTFConfigMaker *m_OMTFConfigMaker;
  XMLConfigWriter *m_Writer;
};

#endif
