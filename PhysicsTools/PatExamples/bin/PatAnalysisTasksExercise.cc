#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerBTag.h"
#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerJEC.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "PhysicsTools/UtilAlgos/interface/FWLiteAnalyzerWrapper.h"

typedef fwlite::AnalyzerWrapper<AnalysisTasksAnalyzerBTag> WrappedFWLiteAnalysisTasksAnalyzerBTag;
typedef fwlite::AnalyzerWrapper<AnalysisTasksAnalyzerJEC> WrappedFWLiteAnalysisTasksAnalyzerJEC;

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

  WrappedFWLiteAnalysisTasksAnalyzerBTag anaBTag(edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process"), std::string("btagAnalyzer"), std::string("analyzeBTag"));
 WrappedFWLiteAnalysisTasksAnalyzerJEC anaJEC(edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process"), std::string("jecAnalyzer"), std::string("analyzeJEC"));
  anaBTag.beginJob();
  anaBTag.analyze();
  anaBTag.endJob();
  anaJEC.beginJob();
  anaJEC.analyze();
  anaJEC.endJob();
  return 0;
}
