#include "sstream"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"
#include "L1Trigger/L1TMuonOverlap/plugins/L1TMuonOverlapParamsESProducer.h"

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapParamsESProducer::L1TMuonOverlapParamsESProducer(const edm::ParameterSet& theConfig){
   //the following line is needed to tell the framework what
   // data is being produced
  setWhatProduced(this, &L1TMuonOverlapParamsESProducer::produceParams);
  
  if (!theConfig.exists("configXMLFile") ) return;
  std::string fName = theConfig.getParameter<edm::FileInPath>("configXMLFile").fullPath();
  
  ///WARNING: filling the CondFormats objects works only for a single XML patterns file.
  if (!theConfig.exists("patternsXMLFiles") ) return;
  std::vector<std::string> fileNames;
  for(auto it: theConfig.getParameter<std::vector<edm::ParameterSet> >("patternsXMLFiles")){
    fileNames.push_back(it.getParameter<edm::FileInPath>("patternsXMLFile").fullPath());
  }  
  
  XMLConfigReader myReader;
  myReader.setConfigFile(fName);
  readConnectionsXML(myReader);
  
  for(auto it: fileNames){
    myReader.setPatternsFile(it);
    readPatternsXML(myReader);
  }

  unsigned int patternsVersion = myReader.getPatternsVersion();
  unsigned int fwVersion =  params.fwVersion();

  params.setFwVersion((fwVersion<<16) + patternsVersion);

}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapParamsESProducer::~L1TMuonOverlapParamsESProducer() {}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapParamsESProducer::readConnectionsXML(const XMLConfigReader & aReader){
  
  aReader.readConfig(&params);
  
  return true;  
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapParamsESProducer::readPatternsXML(XMLConfigReader & aReader){

  l1t::LUT chargeLUT;
  l1t::LUT etaLUT;
  l1t::LUT ptLUT;
  l1t::LUT meanDistPhiLUT;
  l1t::LUT pdfLUT;

  std::vector<l1t::LUT*> luts={ &chargeLUT, &etaLUT, &ptLUT, &meanDistPhiLUT, &pdfLUT};
  std::vector<std::string> types= {"iCharge", "iEta", "iPt", "meanDistPhi", "pdf"};
  aReader.readLUTs(luts,params,types);

  params.setChargeLUT(chargeLUT);
  params.setEtaLUT(etaLUT);
  params.setPtLUT(ptLUT);
  params.setMeanDistPhiLUT(meanDistPhiLUT);
  params.setPdfLUT(pdfLUT);
  
  return true;  
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapParamsESProducer::ReturnType
L1TMuonOverlapParamsESProducer::produceParams(const L1TMuonOverlapParamsRcd& iRecord)
{
   return std::make_unique<L1TMuonOverlapParams>(params);
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsESProducer);

