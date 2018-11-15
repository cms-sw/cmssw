#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDCompareTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/program_options.hpp>
#include <boost/exception/all.hpp>

int main(int argc, char *argv[])
{
  try
  {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  catch (cms::Exception& e)
  {
    edm::LogInfo("DDCompareCPV") << "Attempting to configure the plugin manager. Exception message: " << e.what();
    return 1;
  }
  
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
       "Default is DetectorDescription/RegressionTest/test/configuration.xml")
      ("dist-tolerance,t", boost::program_options::value<std::string>(), "Value tolerance for distances (in mm). "
       "Default value 0.0004 (anything larger is an error)")
      ("rot-tolerance,r", boost::program_options::value<std::string>(), "Value tolerance for rotation matrix elements. "
       "Default value is 0.0004 (anything larger is an error)")
      ("spec-tolerance,s", boost::program_options::value<std::string>(), "Value tolerance for rotation matrix elements. "
       "Default value is 0.0004 (anything larger is an error) NOT USED YET")
      ("user-ns,u", "Use the namespaces in each file and do NOT use the filename as namespace. "
       "Default is to use the filename of each file in the configuration.xml file as a filename")
      ("comp-rot,c", "Use the rotation name when comparing rotations. "
       "Default is to use the matrix only and not the name when comparing DDRotations")
      ("continue-on-error,e", "Continue after an error in values. "
       "Default is to stop at the first error. NOT IMPLEMENTED")
      ("attempt-resync,a", "Continue after an error in graph position, attempt to resync. "
       "Default is to stop at the first mis-match of the graph. NOT IMPLEMENTED");
    
    boost::program_options::positional_options_description p;
    p.add("file1", 1);
    p.add("file2", 2);
    
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
    DDCompOptions ddco;
    bool usrns(false);
    try {
      if (vm.count("file1")) {
	configfile = vm["file1"].as<std::string>();
	if (vm.count("file2")) {
	  configfile2 = vm["file2"].as<std::string>();
	}
      }
      if (vm.count("dist-tolerance"))
	ddco.distTol_ = vm["dist-tolerance"].as<double>();
      if (vm.count("rot-tolerance"))
	ddco.rotTol_ = vm["rot-tolerance"].as<double>();
      if (vm.count("spec-tolerance"))
	ddco.rotTol_ = vm["spec-tolerance"].as<double>();
      if (vm.count("user-ns")) 
	usrns = true;
      if (vm.count("comp-rot")) 
	ddco.compRotName_ = true;
      if (vm.count("continue-on-error")) 
	ddco.contOnError_ = true;
      if (vm.count("attempt-resync"))
	ddco.attResync_ = true;
    }
    catch(boost::exception& e)
    {
      edm::LogInfo("DDCompareCPV") << "Attempting to parse the options. Exception message: " << boost::diagnostic_information(e);
      return 1;
    }

    std::ios_base::fmtflags originalFlags = std::cout.flags();
    
    std::cout << "Settings are: " << std::endl;
    std::cout << "Configuration file 1: " << configfile << std::endl;
    std::cout << "Configuration file 2: " << configfile2 << std::endl;
    std::cout << "Length/distance tolerance: " << ddco.distTol_ << std::endl;
    std::cout << "Rotation matrix element tolerance: " << ddco.rotTol_ << std::endl;
    std::cout << "SpecPar tolerance: " << ddco.specTol_ << std::endl;
    std::cout << "User controlled namespace (both file sets)? " << std::boolalpha << usrns << std::endl;
    std::cout << "Compare Rotation names? " << ddco.compRotName_ << std::endl;
    std::cout << "Continue on error (data mismatch)? " << ddco.contOnError_ << std::endl;
    std::cout << "Attempt resyncronization of disparate graphs? " << ddco.attResync_ << std::endl;

    DDCompactView cpv1( DDName( "CompactView1" ));
    DDLParser myP(cpv1);
    myP.getDDLSAX2FileHandler()->setUserNS(usrns);

    /* The configuration file tells the parser what to parse.
       The sequence of files to be parsed does not matter but for one exception:
       XML containing SpecPar-tags must be parsed AFTER all corresponding
       PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
       files and mention them at end of configuration.xml. Functional SW 
       will not suffer from this restriction).
    */  

    // Use the File-In-Path configuration document provider.
    FIPConfiguration fp(cpv1);
    try
    {
      fp.readConfig(configfile, fullPath);
    }
    catch (cms::Exception& e)
    {
      edm::LogInfo("DDCompareCPV") << "Attempting to read config. Exception message: " << e.what();
      return 1;
    }
    
    std::cout << "FILE 1: " << configfile << std::endl;
    if ( fp.getFileList().empty() ) {
      std::cout << "FILE 1: configuration file has no DDD xml files in it!" << std::endl;
      exit(1);
    }
    int parserResult = myP.parse(fp);
    if (parserResult != 0) {
      std::cout << "FILE 1: problem encountered during parsing. exiting ... " << std::endl;
      exit(1);
    }
    cpv1.lockdown();

    DDCompactView cpv2( DDName( "CompactView2" ));
    DDLParser myP2(cpv2);
    myP2.getDDLSAX2FileHandler()->setUserNS(usrns);

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
    if ( fp2.getFileList().empty() ) {
      std::cout << "FILE 2: configuration file has no DDD xml files in it!" << std::endl;
      exit(1);
    }
    int parserResult2 = myP2.parse(fp2);
    if (parserResult2 != 0) {
      std::cout << "FILE 2: problem encountered during parsing. exiting ... " << std::endl;
      exit(1);
    }
    cpv2.lockdown();

    std::cout << "Parsing completed. Start comparing." << std::endl;

    bool graphmatch = DDCompareCPV(cpv1, cpv2, ddco);

    if (graphmatch) {
      std::cout << "DDCompactView graphs match" << std::endl;
    } else {
      std::cout << "DDCompactView graphs do NOT match" << std::endl;
    }

    // Now set everything back to defaults
    std::cout.flags( originalFlags );

    return 0;
}
