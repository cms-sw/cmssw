#include "DetectorDescription/Core/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

class DDCompactView;

void
DDAlgorithmHandler::initialize( const DDName & algoName,
				const DDLogicalPart & parent,
				const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & mArgs,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & svArgs )
{
  algo_ = std::unique_ptr<DDAlgorithm>(DDAlgorithmFactory::get()->create( algoName.fullname()));
  algo_->setParent( parent );
  algo_->initialize( nArgs, vArgs, mArgs, sArgs, svArgs );
}

void
DDAlgorithmHandler::execute( DDCompactView& cpv )
{
  algo_->execute( cpv );
}
