#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

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
