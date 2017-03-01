#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"

#include <utility>

#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

class DDCompactView;

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

void
DDAlgorithmHandler::initialize( const std::string & algoName,
				const DDLogicalPart & parent,
				const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & mArgs,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & svArgs )
{
  std::pair<std::string,std::string> algoNmNs = DDSplit(algoName);
  algoname_ = algoName;
  algo_ = DDAlgorithmFactory::get()->create(algoname_);
  algo_->setParent(parent);
  algo_->initialize(nArgs,vArgs,mArgs,sArgs, svArgs);
}

void
DDAlgorithmHandler::execute( DDCompactView& cpv )
{
  algo_->execute( cpv );
}
