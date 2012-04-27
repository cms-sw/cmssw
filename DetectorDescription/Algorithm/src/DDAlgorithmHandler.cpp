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

void
DDAlgorithmHandler::initialize( const std::string & algoName,
				const DDLogicalPart & parent,
				const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & mArgs,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & svArgs ) throw ( DDException )
{
  std::pair<std::string,std::string> algoNmNs = DDSplit(algoName);
  algoname_ = algoName;
  DCOUT ('T',"ALGO: name=" + algoNmNs.first + " algo=" + algoName);

  algo_ = DDAlgorithmFactory::get()->create(algoname_);
  algo_->setParent(parent);
  algo_->initialize(nArgs,vArgs,mArgs,sArgs, svArgs);
}

void
DDAlgorithmHandler::execute( DDCompactView& cpv ) throw ( DDException )
{
  algo_->execute( cpv );
}
