#include "DetectorDescription/Core/interface/DDConstant.h"

#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "FWCore/Utilities/interface/Exception.h"

DDConstant::DDConstant()
  : DDBase< DDName, std::unique_ptr<double> >()
{ }

DDConstant::DDConstant( const DDName & name )
  : DDBase< DDName, std::unique_ptr<double> >() 
{
  create( name );
}

DDConstant::DDConstant( const DDName & name, std::unique_ptr<double> vals )
{
  create( name, std::move( vals ));
}  

std::ostream & operator<<(std::ostream & os, const DDConstant & cons)
{
  os << "DDConstant name=" << cons.name(); 
  
  if(cons.isDefined().second) {
    os << " val=" << cons.value();
  }
  else {
    os << " constant is not yet defined, only declared.";
  }  
  return os;
}

void
DDConstant::createConstantsFromEvaluator( ClhepEvaluator& eval )
{
  const auto& vars = eval.variables();
  const auto& vals = eval.values();
  if( vars.size() != vals.size()) {
    throw cms::Exception( "DDException" )
      << "DDConstants::createConstansFromEvaluator(): different size of variable names & values!";
  }
  for( const auto& it : vars ) {
    auto found = it.find( "___" );
    DDName name( std::string( it, found + 3, it.size() - 1 ), std::string( it, 0, found ));       
    DDConstant cst( name, std::make_unique<double>( eval.eval( it.c_str())));
  }
}
