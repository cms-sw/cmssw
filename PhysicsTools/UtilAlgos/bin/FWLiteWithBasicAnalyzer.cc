#include "PhysicsTools/UtilAlgos/interface/BasicMuonAnalyzer.h"
#include "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"

typedef fwlite::AnalyzerWrapper<BasicMuonAnalyzer> WrappedFWLiteMuonAnalyzer;

int main(int argc, char* argv[]) 
{
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();

  // only allow one argument for this simple example which should be the
  // the python cfg file
  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  // get the python configuration
  PythonProcessDesc builder(argv[1]);
  WrappedFWLiteMuonAnalyzer ana(*(builder.processDesc()->getProcessPSet()), std::string("muonAnalyzer"), std::string("analyzeBasicPat"));
  ana.beginJob();
  ana.analyze();
  ana.endJob();
  return 0;
}
