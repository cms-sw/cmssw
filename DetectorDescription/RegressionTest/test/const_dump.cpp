#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
namespace DD { } using namespace DD;

int main(int argc, char *argv[])
{
  std::string const kProgramName = argv[0];
  int rc = 0;

  try {

    // Initialize a DDL Schema aware parser for DDL-documents
    // (DDL ... Detector Description Language)
    cout << "initialize DDL parser" << endl;
    DDCompactView cpv;
    DDLParser myP(cpv);

    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    cout << "about to start parsing" << endl;
    string configfile("DetectorDescription/RegressionTest/test/configuration.xml");
    if (argc==2) {
      configfile = argv[1];
    }

    FIPConfiguration fp(cpv);
    fp.readConfig(configfile);
    int parserResult = myP.parse(fp);
    cout << "done parsing" << std::endl;
    cout.flush();
    if (parserResult != 0) {
      cout << " problem encountered during parsing. exiting ... " << endl;
      exit(1);
    }
    cout << "parsing completed" << endl;
  
    cout << endl << endl << "Start checking!" << endl << endl;
 
    DDErrorDetection ed(cpv);
    ed.report( cpv, std::cout);

    DDConstant::createConstantsFromEvaluator();  // DDConstants are not being created by anyone... it confuses me!
    DDConstant::iterator<DDConstant> cit(DDConstant::begin()), ced(DDConstant::end());
    for(; cit != ced; ++cit) {
      cout << *cit << endl;
    }

    DDVector::iterator<DDVector> vit;
    DDVector::iterator<DDVector> ved(DDVector::end());
    if ( vit == ved ) std::cout << "No DDVectors found." << std::endl;
    for (; vit != ved; ++vit) {
      if (vit->isDefined().second) {
	std::cout << vit->toString() << std::endl;
	const std::vector<double>& tv = *vit;
	std::cout << "size: " << tv.size() << std::endl;
	for (double i : tv) {
	  std::cout << i << "\t";
	}
	std::cout << std::endl;
      }
    }

    std::vector<string> vnames;
    DDVectorGetter::beginWith( "Subdetector", vnames );
    for( std::vector<string>::const_iterator sit = vnames.begin(); sit != vnames.end(); ++sit )
    {
      std::cout << sit->c_str() << std::endl;
    }
    
    return 0;
  }
  //  Deal with any exceptions that may have been thrown.
  catch (cms::Exception& e) {
    std::cout << "cms::Exception caught in "
      	      << kProgramName
	      << "\n"
	      << e.explainSelf();
    rc = 1;
  }
  catch (std::exception& e) {
    std::cout << "Standard library exception caught in "
      	      << kProgramName
	      << "\n"
	      << e.what();
    rc = 1;
  }
  catch (...) {
    std::cout << "Unknown exception caught in "
      	      << kProgramName;
    rc = 2;
  }

  return rc;
}
