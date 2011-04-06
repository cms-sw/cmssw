
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDCompareTools.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

#include <boost/program_options.hpp>

#include <string>
#include <iostream>

int main(int argc, char *argv[])
{
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    // Process the command line arguments
    std::string descString("DDCompareCPV");
    descString += " [options] configurationFileName1 configurationFileName2 Compares two DDCompactViews\n";
    descString += "Allowed options";
    boost::program_options::options_description desc(descString);   
    desc.add_options()
      ("help,h", "Print this help message")
      ("file1,f", boost::program_options::value<std::string>(), "XML configuration file name. "
                 "Default is DetectorDescription/RegressionTest/test/configuration.xml")
      ("file2,g", boost::program_options::value<std::string>(), "XML configuration file name. "
       "Default is DetectorDescription/RegressionTest/test/configuration.xml");
      //      ("path,p", "Specifies filename is a full path and not to use FileInPath to find file. "
      //                 " This option is ignored if a filename is not specified");

    boost::program_options::positional_options_description p;
    p.add("file1", 1);
    p.add("file2", 1);

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
    std::string configfile2("DetectorDescription/RegressionTest/test/configuration.xml");
    if (vm.count("file1")) {
      configfile = vm["file1"].as<std::string>();
      if (vm.count("file2")) {
	configfile2 = vm["file2"].as<std::string>();
      }
//       if (vm.count("path")) {
//         fullPath = true;
//       }
    }

    DDCompactView cpv1;
    DDLParser myP(cpv1);
    myP.getDDLSAX2FileHandler()->setUserNS(false);

    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    // Use the File-In-Path configuration document provider.
    FIPConfiguration fp(cpv1);
    fp.readConfig(configfile, fullPath);
    std::cout << "FILE 1: " << configfile << std::endl;
    int parserResult = myP.parse(fp);
    if (parserResult != 0) {
      std::cout << " problem encountered during parsing file 1. exiting ... " << std::endl;
      exit(1);
    }
    cpv1.lockdown();

    DDCompactView cpv2;
    DDLParser myP2(cpv2);
    myP2.getDDLSAX2FileHandler()->setUserNS(false);

    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    // Use the File-In-Path configuration document provider.
    FIPConfiguration fp2(cpv2);
    fp2.readConfig(configfile2, fullPath);
    std::cout << "FILE 2: " << configfile2 << std::endl;
    int parserResult2 = myP2.parse(fp2);
    if (parserResult2 != 0) {
      std::cout << " problem encountered during parsing file 2. exiting ... " << std::endl;
      exit(1);
    }
    cpv2.lockdown();

    std::cout << "Parsing completed. Start comparing." << std::endl;

//      DDErrorDetection ed(cpv1);

//      bool noErrors = ed.noErrorsInTheReport(cpv1);
//      if (noErrors && fullPath) {
//        std::cout << "DDCompareCPV did not find any errors and is finished." << std::endl;
//      }
//     else {
//       ed.report(cpv1, std::cout);
//       if (!noErrors) {
//         return 1;
//       }
//     }

    // SOMETHING LIKE...
//     const DDCompactView::graph_type& g1=cpv1.graph();
//     const DDCompactView::graph_type& g2=cpv2.graph();
//     DDCompareCPVGraph ddccg;
//     bool graphmatch = ddccg(g1, g2);
    DDCompareCPV ddccpv(true);
    bool graphmatch = ddccpv(cpv1, cpv2);

    if (graphmatch) {
      std::cout << "graphs match" << std::endl;
    } else {
      std::cout << "graphs do NOT match" << std::endl;
    }
    return 0;
}
