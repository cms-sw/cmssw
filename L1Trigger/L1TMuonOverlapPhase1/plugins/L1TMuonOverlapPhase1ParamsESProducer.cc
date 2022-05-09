#include "L1TMuonOverlapPhase1ParamsESProducer.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLConfigReader.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapPhase1ParamsESProducer::L1TMuonOverlapPhase1ParamsESProducer(const edm::ParameterSet& theConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &L1TMuonOverlapPhase1ParamsESProducer::produceParams);

  if (!theConfig.exists("configXMLFile"))
    return;
  std::string fName = theConfig.getParameter<edm::FileInPath>("configXMLFile").fullPath();

  edm::LogVerbatim("OMTFReconstruction") << "L1TMuonOverlapPhase1ParamsESProducer - reading config from file: " << fName
                                         << std::endl;

  if (!theConfig.exists("patternsXMLFiles"))
    return;
  std::vector<std::string> fileNames;
  for (const auto& it : theConfig.getParameter<std::vector<edm::ParameterSet> >("patternsXMLFiles")) {
    std::string fName = it.getParameter<edm::FileInPath>("patternsXMLFile").fullPath();
    fileNames.push_back(fName);
    edm::LogVerbatim("OMTFReconstruction")
        << "L1TMuonOverlapPhase1ParamsESProducer - reading patterns from file: " << fName << std::endl;
  }

  XMLConfigReader myReader;
  myReader.setConfigFile(fName);
  readConnectionsXML(myReader);

  myReader.setPatternsFiles(fileNames);
  readPatternsXML(myReader);

  unsigned int patternsVersion = myReader.getPatternsVersion();
  unsigned int fwVersion = params.fwVersion();

  params.setFwVersion((fwVersion << 16) + patternsVersion);
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapPhase1ParamsESProducer::~L1TMuonOverlapPhase1ParamsESProducer() {}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapPhase1ParamsESProducer::readConnectionsXML(const XMLConfigReader& aReader) {
  aReader.readConfig(&params);

  return true;
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapPhase1ParamsESProducer::readPatternsXML(XMLConfigReader& aReader) {
  l1t::LUT chargeLUT;
  l1t::LUT etaLUT;
  l1t::LUT ptLUT;
  l1t::LUT meanDistPhiLUT;
  l1t::LUT selDistPhiShiftLUT;
  l1t::LUT pdfLUT;

  std::vector<l1t::LUT*> luts = {&chargeLUT, &etaLUT, &ptLUT, &meanDistPhiLUT, &selDistPhiShiftLUT, &pdfLUT};
  std::vector<std::string> types = {"iCharge", "iEta", "iPt", "meanDistPhi", "selDistPhiShift", "pdf"};

  //in the luts we want to have the empty patterns (pt == 0), because it is the way to have the info about the patterns grouping
  aReader.readLUTs(luts, params, types);

  params.setChargeLUT(chargeLUT);
  params.setEtaLUT(etaLUT);
  params.setPtLUT(ptLUT);
  params.setMeanDistPhiLUT(meanDistPhiLUT);
  params.setDistPhiShiftLUT(selDistPhiShiftLUT);
  params.setPdfLUT(pdfLUT);

  return true;
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapPhase1ParamsESProducer::ReturnType L1TMuonOverlapPhase1ParamsESProducer::produceParams(
    const L1TMuonOverlapParamsRcd& iRecord) {
  using namespace edm::es;

  return std::make_shared<L1TMuonOverlapParams>(params);
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapPhase1ParamsESProducer);
