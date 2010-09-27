#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

#include <boost/program_options.hpp>

#include <string>
#include <iostream>

int main(int argc, char *argv[])
{
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    // Process the command line arguments
    std::string descString("DDErrorReport");
    descString += " [options] configurationFileName\n";
    descString += "Allowed options";
    boost::program_options::options_description desc(descString);   
    desc.add_options()
      ("help,h", "Print this help message")
      ("file,f", boost::program_options::value<std::string>(), "XML configuration file name. "
                 "Default is DetectorDescription/RegressionTest/test/configuration.xml")
      ("path,p", "Specifies filename is a full path and not to use FileInPath to find file. "
                 " This option is ignored if a filename is not specified");

    boost::program_options::positional_options_description p;
    p.add("file", -1);

    boost::program_options::variables_map vm;
    try {
      store(boost::program_options::command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
      notify(vm);
    } catch(boost::program_options::error const& iException) {
      std::cerr << "Exception from command line processing: "
                << iException.what() << "\n";
      std::cerr << desc << std::endl;
      return 1;
    }

    if(vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    bool fullPath = false;
    std::string configfile("DetectorDescription/RegressionTest/test/configuration.xml");
    if (vm.count("file")) {
      configfile = vm["file"].as<std::string>();
      if (vm.count("path")) {
        fullPath = true;
      }
    }

    DDCompactView cpv;
    DDLParser myP(cpv);
    myP.getDDLSAX2FileHandler()->setUserNS(false);

    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    // Use the File-In-Path configuration document provider.
    FIPConfiguration fp(cpv);
    fp.readConfig(configfile, fullPath);
    int parserResult = myP.parse(fp);
    if (parserResult != 0) {
      std::cout << " problem encountered during parsing. exiting ... " << std::endl;
      exit(1);
    }
    std::cout << "Parsing completed. Start checking for errors." << std::endl;

    DDErrorDetection ed(cpv);

    bool noErrors = ed.noErrorsInTheReport(cpv);
    if (noErrors && fullPath) {
      std::cout << "DDErrorReport did not find any errors and is finished." << std::endl;
    }
    else {
      ed.report(cpv, std::cout);
      if (!noErrors) {
        return 1;
      }
    }
    return 0;
}
