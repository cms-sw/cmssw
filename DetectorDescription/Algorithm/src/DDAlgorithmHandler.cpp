//
//#include "Utilities/Configuration/interface/Architecture.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDdebug.h"



DDAlgorithmHandler::DDAlgorithmHandler()
: algo_(0)
{
}

DDAlgorithmHandler::~DDAlgorithmHandler()
{
  if (algo_) {
    delete algo_;
  }
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
  //  try {
  //  edmplugin::PluginManager::configure(edmplugin::standard::config());
  //}
  //catch(...) {
  //  std::cout << "FATAL!!! Could not initialize the PluginManager!!!" << std::endl;
  //}

  try {
    //try the old name
    algo_ = DDAlgorithmFactory::get()->create(algoNmNs.first);
  } catch(const cms::Exception& ) { }
  DCOUT ('T',"ALGO: name=" + algoNmNs.first + " algo=" + algoName);
  if (!algo_) {
    try {
      algo_ = DDAlgorithmFactory::get()->create(algoname_);
    }catch(const cms::Exception& e) {
      std::string message("no algorithm with name=\"" + algoName + "\" found. Becauses\n");
      message += e.what();
      throw DDException(message);
    }
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
      //should never get here
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
