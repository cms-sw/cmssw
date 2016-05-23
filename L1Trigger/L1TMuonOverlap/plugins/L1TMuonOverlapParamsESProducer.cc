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
   setWhatProduced(this);

   if (!theConfig.exists("configXMLFile") ) return;
   std::string fName = theConfig.getParameter<edm::FileInPath>("configXMLFile").fullPath();

   ///WARNING: filling the CondFormats objects works only for a single XML patterns file.
   if ( !theConfig.exists("patternsXMLFiles") ) return;
   std::vector<std::string> fileNames;
   for(auto it: theConfig.getParameter<std::vector<edm::ParameterSet> >("patternsXMLFiles")){
     fileNames.push_back(it.getParameter<edm::FileInPath>("patternsXMLFile").fullPath());
   }  

   myOMTFConfig = new OMTFConfiguration(theConfig);   
   XMLConfigReader myReader(myOMTFConfig);

   myReader.setConfigFile(fName);
   readConnectionsXML(&myReader);

   ///Have to fill OMTFConfiguration data to be able to read the Golden Patterns file, as there are
   ///some consistency checks during XML parsing.
   myOMTFConfig->configure(&m_params);

   for(auto it: fileNames){
     myReader.setPatternsFile(it);
     readPatternsXML(&myReader);
   }  
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapParamsESProducer::~L1TMuonOverlapParamsESProducer() {}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapParamsESProducer::readConnectionsXML(XMLConfigReader *aReader){

  aReader->readConfig(&m_params);
  
  return true;
  
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
bool L1TMuonOverlapParamsESProducer::readPatternsXML(XMLConfigReader *aReader){

  l1t::LUT chargeLUT;
  aReader->readLUT(&chargeLUT,"iCharge");
  m_params.setChargeLUT(chargeLUT);

  l1t::LUT etaLUT;
  aReader->readLUT(&etaLUT,"iEta");
  m_params.setEtaLUT(etaLUT);

  l1t::LUT ptLUT;
  aReader->readLUT(&ptLUT,"iPt");
  m_params.setPtLUT(ptLUT);

  l1t::LUT meanDistPhiLUT;
  aReader->readLUT(&meanDistPhiLUT,"meanDistPhi");
  m_params.setMeanDistPhiLUT(meanDistPhiLUT);
  
  l1t::LUT pdfLUT;
  aReader->readLUT(&pdfLUT,"pdf");
  m_params.setPdfLUT(pdfLUT);
  
  return true;  
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
L1TMuonOverlapParamsESProducer::ReturnType
L1TMuonOverlapParamsESProducer::produce(const L1TMuonOverlapParamsRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TMuonOverlapParams> aL1TMTFOverlapParams;
  
   aL1TMTFOverlapParams = boost::shared_ptr<L1TMuonOverlapParams>(new L1TMuonOverlapParams(m_params));
   return aL1TMTFOverlapParams;
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsESProducer);

