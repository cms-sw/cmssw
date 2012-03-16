#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

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
	for (size_t i=0; i < tv.size(); ++i) {
	  std::cout << tv[i] << "\t";
	}
	std::cout << std::endl;
      }
    }
    return 0;
  }
  catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
  {
    cerr << "DDD-PROBLEM:" << endl 
	 << e << endl;
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
