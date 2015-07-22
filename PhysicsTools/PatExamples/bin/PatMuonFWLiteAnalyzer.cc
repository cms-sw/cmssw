#include "PhysicsTools/PatExamples/interface/PatMuonAnalyzer.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"

typedef fwlite::AnalyzerWrapper<PatMuonAnalyzer> PatMuonFWLiteAnalyzer;

int main(int argc, char* argv[]) 
{
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  // only allow one argument for this simple example which should be the
  // the python cfg file
  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }
  if( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ){
    std::cout << " ERROR: ParametersSet 'plot' is missing in your configuration file" << std::endl; exit(0);
  }

  PatMuonFWLiteAnalyzer ana(edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process"), std::string("patMuonAnalyzer"), std::string("patMuonAnalyzer"));
  ana.beginJob();
  ana.analyze();
  ana.endJob();
  return 0;
}
