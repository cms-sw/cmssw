//
//#include "Utilities/Configuration/interface/Architecture.h"
#include "PluginManager/PluginManager.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDSplit.h"



DDAlgorithmHandler::DDAlgorithmHandler()
: algo_(0)
{
}


void DDAlgorithmHandler::initialize(const std::string & algoName,
		  const DDLogicalPart & parent,
		  const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & svArgs) throw (DDException)
{
  std::pair<std::string,std::string> algoNmNs = DDSplit(algoName);
  algoname_ = algoName;
  try {
    seal::PluginManager::get()->initialise();
  }
  catch(...) {
    std::cout << "FATAL!!! Could not initialize the PluginManager!!!" << std::endl;
  }
  algo_ = DDAlgorithmFactory::get()->create(algoNmNs.first);
  std::cout << "ALGO: name=" << algoNmNs.first << " algo=" << algo_ << std::endl;
  if (!algo_) {
    algo_ = DDAlgorithmFactory::get()->create(algoname_);
  }
  if (algo_) {
    try {
      algo_->setParent(parent);
      algo_->initialize(nArgs,vArgs,mArgs,sArgs, svArgs);
    }
    catch(DDException e) {
      throw;
    }
    catch(...) {
      std::string message("initialization of algorithm name=\"" + algoName + "\" failed");
      throw DDException(message);
    }
  }
  else {
    std::string message("no algorithm with name=\"" + algoName + "\" found.");
    throw DDException(message);
  }
}

void DDAlgorithmHandler::execute() throw (DDException)
{
  try {
    algo_->execute();
  }
  catch(const DDException & e) {
    throw;
  }
  catch(...) {
    std::string message("execution of algorithm name=\"" + algoname_ + "\" failed!");
    throw DDException(message);
  }
}
