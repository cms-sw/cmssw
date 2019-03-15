#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include <boost/timer/timer.hpp>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternWithStat.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OmtfName.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessor.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessorTTMerger.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFReconstruction.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLConfigReader.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLConfigWriter.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLEventWriter.h>
#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/PatternGeneratorTT.h>
#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/PatternOptimizer.h>

/*OMTFReconstruction::OMTFReconstruction() :
  m_OMTFConfig(nullptr), m_OMTF(nullptr), aTopElement(nullptr), m_OMTFConfigMaker(nullptr), m_Writer(nullptr){}*/
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::OMTFReconstruction(const edm::ParameterSet& theConfig) :
  m_Config(theConfig), m_OMTFConfig(nullptr), m_OMTF(nullptr), m_OMTFConfigMaker(nullptr) {

  dumpResultToXML = m_Config.getParameter<bool>("dumpResultToXML");
  dumpDetailedResultToXML = m_Config.getParameter<bool>("dumpDetailedResultToXML");
  //m_Config.getParameter<std::string>("XMLDumpFileName");  
  bxMin = m_Config.exists("bxMin") ? m_Config.getParameter<int>("bxMin") : 0;
  bxMax = m_Config.exists("bxMax") ? m_Config.getParameter<int>("bxMax") : 0;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::~OMTFReconstruction(){
  delete m_OMTFConfig;

  //if (m_Writer) delete m_Writer;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginJob() {
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<"test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    m_OMTFConfig = new OMTFConfiguration();
    //m_OMTF = new OMTFProcessor<GoldenPattern>();
    //m_OMTF = new OMTFProcessor<GoldenPatternParametrised>();

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::endJob(){
/*  if(dumpResultToXML){
    std::string fName = m_Config.getParameter<std::string>("XMLDumpFileName");
    m_Writer->finaliseXMLDocument(fName);
  } */

  for(auto& obs : observers) {
    obs->endJob();
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  const L1TMuonOverlapParams* omtfParams = 0;

  std::string processorType = "OMTFProcessor"; //GoldenPatternParametrised GoldenPatternWithStat GoldenPattern
  if(m_Config.exists("processorType") ){
    processorType = m_Config.getParameter<std::string>("processorType");
  }

  if(m_OMTF == 0 || m_Config.exists("patternsXMLFile") == false) {
    edm::LogImportant("OMTFReconstruction") << "retrieving parameters from Event Setup" << std::endl;

    const L1TMuonOverlapParamsRcd& omtfRcd = iSetup.get<L1TMuonOverlapParamsRcd>();
    edm::ESHandle<L1TMuonOverlapParams> omtfParamsHandle;
    omtfRcd.get(omtfParamsHandle);
    omtfParams = omtfParamsHandle.product();
    if (!omtfParams) {
      edm::LogError("OMTFReconstruction") << "Could not retrieve parameters from Event Setup" << std::endl;
    }
    m_OMTFConfig->configure(omtfParams);
    //patterns from the L1TMuonBayesOmtfParamsESProducer, are reloaded every run begin
    if(m_OMTF == 0 && m_Config.exists("patternsXMLFile") == false) {
      edm::LogImportant("OMTFReconstruction") << "getting patterns from L1TMuonBayesOmtfParamsESProducer" << std::endl;
      //m_OMTFConfig->initPatternPtRange();
      if(processorType == "OMTFProcessor")
        m_OMTF.reset(new OMTFProcessor<GoldenPattern>(m_OMTFConfig, m_Config, iSetup, omtfParams) );
      else if(processorType == "OMTFProcessorTTMerger")
        m_OMTF.reset(new OMTFProcessorTTMerger<GoldenPattern>(m_OMTFConfig, m_Config, iSetup, omtfParams) );
    }
  }
  if(m_OMTF == 0 && m_Config.exists("patternsXMLFile") ) {//if we read the patterns directly from the xml, we do it only once, at the beginning of the first run, not every run
    std::string patternsXMLFile = m_Config.getParameter<edm::FileInPath>("patternsXMLFile").fullPath();
    edm::LogImportant("OMTFReconstruction") << "reading patterns from "<<patternsXMLFile << std::endl;
    XMLConfigReader xmlReader;
    xmlReader.setPatternsFile(patternsXMLFile);

    std::string patternType = "GoldenPattern"; //GoldenPatternParametrised GoldenPatternWithStat GoldenPattern
    if(m_Config.exists("patternType") ){
      patternType = m_Config.getParameter<std::string>("patternType");
    }

    std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
    if(patternType == "GoldenPattern") {
      auto const& gps = xmlReader.readPatterns<GoldenPattern>(*omtfParams);

      if(processorType == "OMTFProcessor") {
        m_OMTF.reset(new OMTFProcessor<GoldenPattern>(m_OMTFConfig, m_Config, iSetup, gps) );
      }
      else if(processorType == "OMTFProcessorTTMerger") {
        m_OMTF.reset(new OMTFProcessorTTMerger<GoldenPattern>(m_OMTFConfig, m_Config, iSetup, gps) );
      }

      edm::LogImportant("OMTFReconstruction") << "OMTFProcessor constructed. processorType "<<processorType<<". GoldenPattern type: "<<patternType<<" size: "<<gps.size() << std::endl;

      for(auto& gp : gps) {
        edm::LogImportant("OMTFReconstruction")<<gp->key()<<" "
            <<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptFrom
            <<" - "<<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptTo<<" GeV"<<std::endl;
      }
    }
    else if(patternType == "GoldenPatternWithStat") {
      std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;

      auto gps = xmlReader.readPatterns<GoldenPatternWithStat>(*omtfParams);

      if(processorType == "OMTFProcessor") {
        std::unique_ptr<IOMTFEmulationObserver> obs(new PatternOptimizer(m_Config, m_OMTFConfig, gps));
        m_OMTF.reset(new OMTFProcessor<GoldenPatternWithStat>(m_OMTFConfig, m_Config, iSetup, gps) );
        observers.emplace_back(std::move(obs));
      }
      else if(processorType == "OMTFProcessorTTMerger") {
        std::unique_ptr<IOMTFEmulationObserver> obs(new PatternGeneratorTT(m_Config, m_OMTFConfig, gps));
        observers.emplace_back(std::move(obs));
        m_OMTF.reset(new OMTFProcessorTTMerger<GoldenPatternWithStat>(m_OMTFConfig, m_Config, iSetup, gps) );
      }
      edm::LogImportant("OMTFReconstruction") << "OMTFProcessor constructed. GoldenPattern type: "<<patternType<<" size: "<<gps.size() << std::endl;

      for(auto& gp : gps) {
        edm::LogImportant("OMTFReconstruction")<<gp->key()<<" "
            <<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptFrom
            <<" - "<<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptTo<<" GeV"<<std::endl;
      }
    }
    else if(patternType == "GoldenPatternWithThresh") {
      std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
      auto gps = xmlReader.readPatterns<GoldenPatternWithThresh>(*omtfParams);
      m_OMTF.reset(new OMTFProcessor<GoldenPatternWithThresh>(m_OMTFConfig, m_Config, iSetup, gps) );
      edm::LogImportant("OMTFReconstruction") << "OMTFProcessor constructed. GoldenPattern type: "<<patternType<<" size: "<<gps.size() << std::endl;

      //std::unique_ptr<IOMTFEmulationObserver> obs(new PatternOptimizer(m_Config, m_OMTFConfig, gps));
      //observers.emplace_back(std::move(obs));

      for(auto& gp : gps) {
        edm::LogImportant("OMTFReconstruction")<<gp->key()<<" "
            <<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptFrom
            <<" - "<<m_OMTFConfig->getPatternPtRange(gp->key().theNumber).ptTo<<" GeV"<<" threshold "<<gp->getThreshold(0)<<std::endl;
      }
    }
/*    else if(patternType == "GoldenPatternParametrised") {
      auto const& gps = xmlReader.readPatterns<GoldenPattern>(*omtfParams);

      OMTFProcessor<GoldenPatternParametrised>::GoldenPatternVec gpsParametrised;
      for(auto& gp :  gps) {
        if(gp.get() != 0) {
          gp->setConfig(m_OMTFConfig);
          edm::LogImportant("OMTFReconstruction") <<gp->key()<< std::endl;
          GoldenPatternParametrised* newGp = new GoldenPatternParametrised(gp.get());
          gpsParametrised.emplace_back(newGp);
        }
      }

      m_OMTF.reset(new OMTFProcessor<GoldenPatternParametrised>(m_OMTFConfig, m_Config, iSetup, gpsParametrised) );

      edm::LogImportant("OMTFReconstruction") << "OMTFProcessor constructed. GoldenPattern type: "<<patternType<<" size: "<<gps.size() << std::endl;
    }*/
    else {
      throw cms::Exception("OMTFReconstruction::beginRun: unknown GoldenPattern type: " + patternType);
    }

  }

  if(dumpResultToXML){
    std::unique_ptr<IOMTFEmulationObserver> obs(new XMLEventWriter(m_OMTFConfig, m_Config.getParameter<std::string>("XMLDumpFileName")));
    observers.emplace_back(std::move(obs));
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
std::unique_ptr<l1t::RegionalMuonCandBxCollection> OMTFReconstruction::reconstruct(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  LogTrace("omtfEventPrintout")<<"\n"<<__FUNCTION__<<":"<<__LINE__<<" iEvent "<<iEvent.id().event()<<endl;
  m_OMTF->loadAndFilterDigis(iEvent, m_Config);

  //if(dumpResultToXML) aTopElement = m_Writer->writeEventHeader(iEvent.id().event());
  theEvent = iEvent.id().event();
  
  for(auto& obs : observers) {
    obs->observeEventBegin(iEvent);
  }

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> candidates(new l1t::RegionalMuonCandBxCollection);
  candidates->setBXRange(bxMin, bxMax);


  ///The order is important: first put omtf_pos candidates, then omtf_neg.
  for(int bx = bxMin; bx<= bxMax; bx++) {
  
    for(unsigned int iProcessor=0; iProcessor<m_OMTFConfig->nProcessors(); ++iProcessor) {
      std::vector<l1t::RegionalMuonCand> candMuons = m_OMTF->run(iProcessor, l1t::tftype::omtf_pos, bx, observers);

      //fill outgoing collection
      for (auto & candMuon :  candMuons) {
        candidates->push_back(bx, candMuon);
      }
    }

    for(unsigned int iProcessor=0; iProcessor<m_OMTFConfig->nProcessors(); ++iProcessor) {
      std::vector<l1t::RegionalMuonCand> candMuons = m_OMTF->run(iProcessor, l1t::tftype::omtf_neg, bx, observers);

      //fill outgoing collection
      for (auto & candMuon :  candMuons) {
        candidates->push_back(bx, candMuon);
      }
    }

    edm::LogInfo("OMTFReconstruction") <<"OMTF:  Number of candidates in BX="<<bx<<": "<<candidates->size(bx) << std::endl;;  
  }
  
  for(auto& obs : observers) {
    obs->observeEventEnd(iEvent);
  }

  return candidates;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/*void OMTFReconstruction::loadAndFilterDigis(const edm::Event& iEvent){

  // Filter digis by dropping digis from selected (by cfg.py) subsystems
  if(!m_Config.getParameter<bool>("dropDTPrimitives")){
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTPh"),dtPhDigis);
    iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcDTTh"),dtThDigis);
  }
  if(!m_Config.getParameter<bool>("dropRPCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcRPC"),rpcDigis);
  if(!m_Config.getParameter<bool>("dropCSCPrimitives")) iEvent.getByLabel(m_Config.getParameter<edm::InputTag>("srcCSC"),cscDigis);
  
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::getProcessorCandidates(unsigned int iProcessor, l1t::tftype mtfType, int bx,
               l1t::RegionalMuonCandBxCollection & omtfCandidates){

  //boost::timer::auto_cpu_timer t("%ws wall, %us user in getProcessorCandidates\n");
  m_InputMaker.setFlag(0);
  OMTFinput input = m_InputMaker.buildInputForProcessor(dtPhDigis.product(),
                dtThDigis.product(),
                cscDigis.product(),
                rpcDigis.product(),
                iProcessor, mtfType, bx);
  int flag = m_InputMaker.getFlag();
  //cout<<"buildInputForProce "; t.report();
  m_OMTF->processInput(iProcessor, mtfType, input);
  //cout<<"processInput       "; t.report();
  std::vector<AlgoMuon> algoCandidates =  m_OMTF->sortResults(iProcessor, mtfType);
  //cout<<"sortResults        "; t.report();
  // perform GB 
  std::vector<AlgoMuon> gbCandidates =  m_OMTF->ghostBust(algoCandidates);
  //cout<<"ghostBust          "; t.report();
  // fill RegionalMuonCand colleciton
  std::vector<l1t::RegionalMuonCand> candMuons = m_OMTF->getFinalcandidates(iProcessor, mtfType, gbCandidates);
  //cout<<"getFinalcandidates "; t.report();
  //fill outgoing collection
  for (auto & candMuon :  candMuons) {
     candMuon.setHwQual( candMuon.hwQual() | flag);         //FIXME temporary debug fix
     omtfCandidates.push_back(bx, candMuon);
  }

  //dump to XML
  //if(bx==0) writeResultToXML(iProcessor, mtfType,  input, algoCandidates, candMuons); //TODO handle bx
  //if(bx==0)
  for(auto& obs : observers) {
    obs->observeProcesorEmulation(iProcessor, mtfType,  input, algoCandidates, gbCandidates, candMuons);
  }
}*/
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
