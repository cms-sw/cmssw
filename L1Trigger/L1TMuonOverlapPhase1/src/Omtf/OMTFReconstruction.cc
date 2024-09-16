#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFReconstruction.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/ProcessorBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLConfigReader.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLEventWriter.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/ProcConfigurationBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/CandidateSimMuonMatcher.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/DataROOTDumper2.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/EventCapture.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternGenerator.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Common/interface/EventBase.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::OMTFReconstruction(const edm::ParameterSet& parameterSet, MuStubsInputTokens& muStubsInputTokens)
    : edmParameterSet(parameterSet),
      muStubsInputTokens(muStubsInputTokens),
      omtfConfig(new OMTFConfiguration()),
      omtfProc(nullptr),
      m_OMTFConfigMaker(nullptr) {
  bxMin = edmParameterSet.exists("bxMin") ? edmParameterSet.getParameter<int>("bxMin") : 0;
  bxMax = edmParameterSet.exists("bxMax") ? edmParameterSet.getParameter<int>("bxMax") : 0;

  edm::LogVerbatim("OMTFReconstruction") << "running emulation for the bxMin " << bxMin << " - bxMax " << bxMax
                                         << std::endl;
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFReconstruction::~OMTFReconstruction() {}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginJob() {
  inputMaker = std::make_unique<OMTFinputMaker>(
      edmParameterSet, muStubsInputTokens, omtfConfig.get(), std::make_unique<OmtfAngleConverter>());
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::endJob() {
  for (auto& obs : observers) {
    obs->endJob();
  }
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFReconstruction::beginRun(edm::Run const& run,
                                  edm::EventSetup const& eventSetup,
                                  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd>& omtfParamsEsToken,
                                  const MuonGeometryTokens& muonGeometryTokens,
                                  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
                                  const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) {
  std::string processorType = "OMTFProcessor";  //GoldenPatternWithStat GoldenPattern
  if (edmParameterSet.exists("processorType")) {
    processorType = edmParameterSet.getParameter<std::string>("processorType");
  }

  bool buildPatternsFromXml = (edmParameterSet.exists("patternsXMLFile") || edmParameterSet.exists("patternsXMLFiles"));

  bool readConfigFromXml = edmParameterSet.exists("configXMLFile");

  if (buildPatternsFromXml != readConfigFromXml)
    throw cms::Exception(
        "OMTFReconstruction::beginRun: buildPatternsFromXml != readConfigFromXml -  both patternsXMLFiles and "
        "configXMLFile should be defined (or not) for the simOmtDigis or simOmtfPhase2Digis");

  edm::LogVerbatim("OMTFReconstruction") << "OMTFReconstruction::beginRun " << run.id()
                                         << " buildPatternsFromXml: " << buildPatternsFromXml << std::endl;

  //if the buildPatternsFromXml == false - we are making the omtfConfig and omtfProc for every run,
  //as the configuration my change between the runs,
  if (buildPatternsFromXml == false) {
    if (omtfParamsRecordWatcher.check(eventSetup)) {
      edm::LogVerbatim("OMTFReconstruction") << "retrieving omtfParams from EventSetup" << std::endl;

      const L1TMuonOverlapParams* omtfParamsFromES = &(eventSetup.getData(omtfParamsEsToken));
      if (!omtfParamsFromES) {
        edm::LogError("OMTFReconstruction") << "Could not retrieve omtfParams from Event Setup" << std::endl;
        throw cms::Exception("OMTFReconstruction::beginRun: Could not retrieve omtfParams from Event Setup");
      }

      omtfConfig->configure(omtfParamsFromES);

      //the parameters can be overwritten from the python config
      omtfConfig->configureFromEdmParameterSet(edmParameterSet);

      inputMaker->initialize(edmParameterSet, eventSetup, muonGeometryTokens);

      //patterns from the edm::EventSetup are reloaded every beginRun
      //therefore OMTFProcessor is re-created here
      edm::LogVerbatim("OMTFReconstruction") << "getting patterns from EventSetup" << std::endl;
      if (processorType == "OMTFProcessor") {
        omtfProc = std::make_unique<OMTFProcessor<GoldenPattern> >(
            omtfConfig.get(), edmParameterSet, eventSetup, omtfParamsFromES);
        omtfProc->printInfo();
      }
    }
  }

  //if buildPatternsFromXml == true - the entire configuration (patterns and hwToLogicLayer) comes from phyton,
  //so we read it only once, at the beginning of the first run, not every run
  if (omtfProc == nullptr && buildPatternsFromXml) {
    std::string fName = edmParameterSet.getParameter<edm::FileInPath>("configXMLFile").fullPath();

    edm::LogVerbatim("OMTFReconstruction")
        << "OMTFReconstruction::beginRun - reading config from file: " << fName << std::endl;

    XMLConfigReader xmlConfigReader;
    xmlConfigReader.setConfigFile(fName);

    omtfParams.reset(new L1TMuonOverlapParams());
    xmlConfigReader.readConfig(omtfParams.get());

    //getPatternsVersion() parses the entire patterns xml - si it is very inefficient
    //moreover, PatternsVersion is not used anywhere
    //Therefore we we dont use xmlPatternReader.getPatternsVersion(); but set patternsVersion to 0
    unsigned int patternsVersion = 0;
    unsigned int fwVersion = omtfParams->fwVersion();
    omtfParams->setFwVersion((fwVersion << 16) + patternsVersion);

    omtfConfig->configure(omtfParams.get());

    //the parameters can be overwritten from the python config
    omtfConfig->configureFromEdmParameterSet(edmParameterSet);

    inputMaker->initialize(edmParameterSet, eventSetup, muonGeometryTokens);

    //reading patterns from the xml----------------------------------------------------------
    std::vector<std::string> patternsXMLFiles;

    if (edmParameterSet.exists("patternsXMLFile")) {
      patternsXMLFiles.push_back(edmParameterSet.getParameter<edm::FileInPath>("patternsXMLFile").fullPath());
    } else if (edmParameterSet.exists("patternsXMLFiles")) {
      for (const auto& it : edmParameterSet.getParameter<std::vector<edm::ParameterSet> >("patternsXMLFiles")) {
        patternsXMLFiles.push_back(it.getParameter<edm::FileInPath>("patternsXMLFile").fullPath());
      }
    }

    for (auto& patternsXMLFile : patternsXMLFiles)
      edm::LogVerbatim("OMTFReconstruction") << "reading patterns from " << patternsXMLFile << std::endl;

    std::string patternType = "GoldenPattern";  //GoldenPatternWithStat GoldenPattern
    if (edmParameterSet.exists("patternType")) {
      patternType = edmParameterSet.getParameter<std::string>("patternType");
    }

    if (patternType == "GoldenPattern") {
      if (processorType == "OMTFProcessor") {
        if (omtfParams) {
          omtfProc = std::make_unique<OMTFProcessor<GoldenPattern> >(
              omtfConfig.get(),
              edmParameterSet,
              eventSetup,
              xmlConfigReader.readPatterns<GoldenPattern>(*omtfParams, patternsXMLFiles, false));
        } else {  //in principle should not happen
          throw cms::Exception("OMTFReconstruction::beginRun: omtfParams is nullptr");
        }
      }

      edm::LogVerbatim("OMTFReconstruction") << "OMTFProcessor constructed. processorType " << processorType
                                             << ". GoldenPattern type: " << patternType << std::endl;
    } else if (patternType == "GoldenPatternWithStat") {
      //pattern generation is only possible if the processor is constructed only once per job
      //PatternGenerator modifies the patterns!!!
      if (processorType == "OMTFProcessor") {
        if (omtfParams) {
          omtfProc = std::make_unique<OMTFProcessor<GoldenPatternWithStat> >(
              omtfConfig.get(),
              edmParameterSet,
              eventSetup,
              xmlConfigReader.readPatterns<GoldenPatternWithStat>(*omtfParams, patternsXMLFiles, false));
        } else {  //in principle should not happen
          throw cms::Exception("OMTFReconstruction::beginRun: omtfParams is nullptr");
        }
      }
    } else {
      throw cms::Exception("OMTFReconstruction::beginRun: unknown GoldenPattern type: " + patternType);
    }

    omtfProc->printInfo();
  }

  addObservers(muonGeometryTokens, magneticFieldEsToken, propagatorEsToken);

  for (auto& obs : observers) {
    obs->beginRun(eventSetup);
  }
}

void OMTFReconstruction::addObservers(
    const MuonGeometryTokens& muonGeometryTokens,
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>& magneticFieldEsToken,
    const edm::ESGetToken<Propagator, TrackingComponentsRecord>& propagatorEsToken) {
  if (!observers.empty())  //assuring it is done only at the first run
    return;

  edm::LogVerbatim("OMTFReconstruction") << "OMTFReconstruction::addObservers " << std::endl;

  //omtfConfig is created at constructor, and is not re-created at the the start of the run, so this is OK
  if (edmParameterSet.exists("dumpResultToXML")) {
    if (edmParameterSet.getParameter<bool>("dumpResultToXML"))
      observers.emplace_back(std::make_unique<XMLEventWriter>(
          omtfConfig.get(), edmParameterSet.getParameter<std::string>("XMLDumpFileName")));
  }

  CandidateSimMuonMatcher* candidateSimMuonMatcher = nullptr;

  if (edmParameterSet.exists("candidateSimMuonMatcher")) {
    if (edmParameterSet.getParameter<bool>("candidateSimMuonMatcher")) {
      observers.emplace_back(std::make_unique<CandidateSimMuonMatcher>(
          edmParameterSet, omtfConfig.get(), magneticFieldEsToken, propagatorEsToken));
      candidateSimMuonMatcher = static_cast<CandidateSimMuonMatcher*>(observers.back().get());
    }
  }

  auto omtfProcGoldenPat = dynamic_cast<OMTFProcessor<GoldenPattern>*>(omtfProc.get());
  if (omtfProcGoldenPat) {
    if (edmParameterSet.exists("eventCaptureDebug"))
      if (edmParameterSet.getParameter<bool>("eventCaptureDebug")) {
        observers.emplace_back(std::make_unique<EventCapture>(
            edmParameterSet, omtfConfig.get(), candidateSimMuonMatcher, muonGeometryTokens
            //&(omtfProcGoldenPat->getPatterns() ),
            //watch out, will crash if the proc is re-constructed from the DB after L1TMuonOverlapParamsRcd change
            ));
      }

    if (edmParameterSet.exists("dumpHitsToROOT") && edmParameterSet.getParameter<bool>("dumpHitsToROOT")) {
      //std::string rootFileName = edmParameterSet.getParameter<std::string>("dumpHitsFileName");
      if (candidateSimMuonMatcher == nullptr) {
        edm::LogVerbatim("OMTFReconstruction")
            << "dumpHitsToROOT needs candidateSimMuonMatcher, but it is null " << std::endl;
        throw cms::Exception("dumpHitsToROOT needs candidateSimMuonMatcher, but it is null");
      }
      observers.emplace_back(
          std::make_unique<DataROOTDumper2>(edmParameterSet, omtfConfig.get(), candidateSimMuonMatcher));
    }
  }

  auto omtfProcGoldenPatWithStat = dynamic_cast<OMTFProcessor<GoldenPatternWithStat>*>(omtfProc.get());
  if (omtfProcGoldenPatWithStat) {
    if (edmParameterSet.exists("eventCaptureDebug"))
      if (edmParameterSet.getParameter<bool>("eventCaptureDebug")) {
        observers.emplace_back(std::make_unique<EventCapture>(
            edmParameterSet, omtfConfig.get(), candidateSimMuonMatcher, muonGeometryTokens
            //&(omtfProcGoldenPat->getPatterns() ),
            //watch out, will crash if the proc is re-constructed from the DB after L1TMuonOverlapParamsRcd change
            ));
      }

    if (edmParameterSet.exists("generatePatterns") && edmParameterSet.getParameter<bool>("generatePatterns")) {
      observers.emplace_back(std::make_unique<PatternGenerator>(
          edmParameterSet, omtfConfig.get(), omtfProcGoldenPatWithStat->getPatterns(), candidateSimMuonMatcher));
      edm::LogVerbatim("OMTFReconstruction") << "generatePatterns: true " << std::endl;
    }
  }
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
std::unique_ptr<l1t::RegionalMuonCandBxCollection> OMTFReconstruction::reconstruct(const edm::Event& iEvent,
                                                                                   const edm::EventSetup& evSetup) {
  LogTrace("l1tOmtfEventPrint") << "\n" << __FUNCTION__ << ":" << __LINE__ << " iEvent " << iEvent.id().event() << endl;
  inputMaker->loadAndFilterDigis(iEvent);

  for (auto& obs : observers) {
    obs->observeEventBegin(iEvent);
  }

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> candidates = std::make_unique<l1t::RegionalMuonCandBxCollection>();
  candidates->setBXRange(bxMin, bxMax);

  ///The order is important: first put omtf_pos candidates, then omtf_neg.
  for (int bx = bxMin; bx <= bxMax; bx++) {
    for (unsigned int iProcessor = 0; iProcessor < omtfConfig->nProcessors(); ++iProcessor) {
      std::vector<l1t::RegionalMuonCand> candMuons =
          omtfProc->run(iProcessor, l1t::tftype::omtf_pos, bx, inputMaker.get(), observers);

      //fill outgoing collection
      for (auto& candMuon : candMuons) {
        candidates->push_back(bx, candMuon);
      }
    }

    for (unsigned int iProcessor = 0; iProcessor < omtfConfig->nProcessors(); ++iProcessor) {
      std::vector<l1t::RegionalMuonCand> candMuons =
          omtfProc->run(iProcessor, l1t::tftype::omtf_neg, bx, inputMaker.get(), observers);

      //fill outgoing collection
      for (auto& candMuon : candMuons) {
        candidates->push_back(bx, candMuon);
      }
    }

    //edm::LogInfo("OMTFReconstruction") <<"OMTF:  Number of candidates in BX="<<bx<<": "<<candidates->size(bx) << std::endl;;
  }

  for (auto& obs : observers) {
    obs->observeEventEnd(iEvent, candidates);
  }

  return candidates;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
