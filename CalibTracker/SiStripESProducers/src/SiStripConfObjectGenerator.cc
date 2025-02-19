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

void SiStripConfObjectGenerator::createObject()
{
  parameters_ = _pset.getParameter<std::vector<edm::ParameterSet> >("Parameters");
  std::vector<edm::ParameterSet>::const_iterator parIt = parameters_.begin();
  obj_ = new SiStripConfObject();
  for( ; parIt != parameters_.end(); ++parIt ) {
    if( parIt->getParameter<std::string>("ParameterType") == "int" ) {
      obj_->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<int32_t>("ParameterValue"));
    }
    else if( parIt->getParameter<std::string>("ParameterType") == "double" ) {
      obj_->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<double>("ParameterValue"));
    }
    if( parIt->getParameter<std::string>("ParameterType") == "string" ) {
      obj_->put(parIt->getParameter<std::string>("ParameterName"), parIt->getParameter<std::string>("ParameterValue"));
    }
  }
}
