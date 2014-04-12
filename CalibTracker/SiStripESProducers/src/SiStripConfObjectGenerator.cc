#include "CalibTracker/SiStripESProducers/interface/SiStripConfObjectGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripConfObjectGenerator::SiStripConfObjectGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripConfObject>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripConfObjectGenerator") << "[SiStripConfObjectGenerator::SiStripConfObjectGenerator]";
}

SiStripConfObjectGenerator::~SiStripConfObjectGenerator()
{
  edm::LogInfo("SiStripConfObjectGenerator") << "[SiStripConfObjectGenerator::~SiStripConfObjectGenerator]";
}

SiStripConfObject* SiStripConfObjectGenerator::createObject()
{
  parameters_ = _pset.getParameter<std::vector<edm::ParameterSet> >("Parameters");
  std::vector<edm::ParameterSet>::const_iterator parIt = parameters_.begin();
  SiStripConfObject* obj = new SiStripConfObject();
  for( ; parIt != parameters_.end(); ++parIt ) {
    if( parIt->getParameter<std::string>("ParameterType") == "int" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<int32_t>("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "double" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<double>("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "string" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<std::string>("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "bool" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<bool>("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "vint32" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<std::vector<int> >("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "vstring" ) {
      obj->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<std::vector<std::string> >("ParameterValue"));
    }
  }
  return obj;
}
