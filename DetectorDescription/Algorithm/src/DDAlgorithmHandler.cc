#include "DetectorDescription/Algorithm/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

class DDCompactView;

DDAlgorithmHandler::DDAlgorithmHandler( void )
{}

DDAlgorithmHandler::~DDAlgorithmHandler( void )
{}

void
DDAlgorithmHandler::initialize( const std::string & algoName,
				const DDLogicalPart & parent,
				const DDNumericArguments & nArgs,
				const DDVectorArguments & vArgs,
				const DDMapArguments & mArgs,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & svArgs )
{
  algoname_ = algoName;
  algo_ = std::unique_ptr<DDAlgorithm>( DDAlgorithmFactory::get()->create( algoname_ ));
  algo_->setParent( parent );
  algo_->initialize( nArgs, vArgs, mArgs, sArgs, svArgs );
}

void
DDAlgorithmHandler::execute( DDCompactView& cpv )
{
  algo_->execute( cpv );
}
