#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include <boost/shared_ptr.hpp>
// #include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
// #include "FWCore/Utilities/interface/Presence.h"
// #include "FWCore/PluginManager/interface/PresenceFactory.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
// #include "FWCore/ServiceRegistry/interface/Service.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace DD { } using namespace DD;

int main(int argc, char *argv[])
{
  std::string const kProgramName = argv[0];
  edmplugin::PluginManager::configure(edmplugin::standard::config());
    std::cout << "initialize DDL parser" << std::endl;
    DDCompactView cpv;
    DDLParser myP(cpv);// = DDLParser::instance();
    myP.getDDLSAX2FileHandler()->setUserNS(false);

    //     std::cout << "about to set configuration" << std::endl;
    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    std::cout << "about to start parsing" << std::endl;
    std::string configfile("DetectorDescription/RegressionTest/test/configuration.xml");
    if (argc==2) {
      configfile = argv[1];
    } else {
      std::cout << "running default " << configfile << std::endl;
    }
    //    Use the File-In-Path configuration document provider.
    FIPConfiguration fp(cpv);
    fp.readConfig(configfile);
    int parserResult = myP.parse(fp);
    std::cout << "done parsing" << std::endl;
    std::cout.flush();
    if (parserResult != 0) {
      std::cout << " problem encountered during parsing. exiting ... " << std::endl;
      exit(1);
    }
    std::cout << " parsing completed" << std::endl;
    std::cout << std::endl << std::endl << "Start checking!" << std::endl << std::endl;
    std::cout.flush();

    DDErrorDetection ed(cpv);
    // maybe later   ed.report(cpv,std::cout);
    //    DDErrorDetection ed;
    //    ed.scan();
    ed.report(cpv, std::cout);

    return 0;
  
}
