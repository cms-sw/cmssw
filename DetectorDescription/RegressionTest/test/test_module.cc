#include <iostream>
#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

class DDCompactView;

class DDTestAlgorithm : public DDAlgorithm
{
public:
  DDTestAlgorithm( void ) {}
  virtual ~DDTestAlgorithm( void ){}
 
  void initialize( const DDNumericArguments &,
		   const DDVectorArguments &,
		   const DDMapArguments &,
		   const DDStringArguments &,
		   const DDStringVectorArguments & )
  {
    std::cout << "DDTestAlgorithm::initialize\n";
  }

  void execute( DDCompactView& ) {
    std::cout << "DDTestAlgorithm::execute\n";
  }
};

DEFINE_EDM_PLUGIN( DDAlgorithmFactory, DDTestAlgorithm, "test:DDTestAlgorithm");
