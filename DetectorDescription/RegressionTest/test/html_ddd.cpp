#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "This program does nothing unless it is given a command line argument.\n";
    std::cout << "The argument should be the name of an xml configuration file.\n";
    return 0;
  }

  // required for main() in cmssw framework
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  try {   
    // Initialize a DDL Schema aware parser for DDL-documents
    // (DDL ... Detector Description Language)
    std::cout << "initialize DDL parser" << std::endl;
    DDCompactView ddcpv;
    DDLParser myP(ddcpv);
    myP.getDDLSAX2FileHandler()->setUserNS(false);

    std::cout << "about to set configuration" << std::endl;
    //  std::string configfile("configuration.xml");
    std::string configfile("DetectorDescription/RegressionTest/test/");
    if (argc==2) {
      configfile += argv[1];
    } else {
      configfile += "configuration.xml";
    }
    std::cout << configfile << std::endl;
    FIPConfiguration fp(ddcpv);
    fp.readConfig(configfile);
    fp.dumpFileList();
    std::cout << "about to start parsing" << std::endl;
    int parserResult = myP.parse(fp);

    if (parserResult != 0) {
      std::cout << " problem encountered during parsing. exiting ... " << std::endl;
      exit(1);
    }
    std::cout << " parsing completed" << std::endl;
  
    std::cout << std::endl << std::endl << "Start checking!" << std::endl << std::endl;
  
    DDCheck(ddcpv, std::cout);

    std::cout << std::endl << "Done with DDCheck!" << std::endl << std::endl;  
  
    typedef DDHtmlFormatter::ns_type ns_type;
  
    ns_type names;
    { 
      DDLogicalPart lp;
      findNameSpaces(lp, names);
    }
  
    std::cout << names.size() << " namespaces found: " << std::endl << std::endl;
    ns_type::const_iterator nit = names.begin();
    for(; nit != names.end(); ++nit) {
      std::cout << nit->first 
		<< " has " 
		<< nit->second.size()
		<< " entries." << std::endl; 
    }
 
    DDErrorDetection ed(ddcpv);

    ed.report(ddcpv, std::cout);
  
    DDHtmlLpDetails lpd("lp","LogicalParts");
    dd_to_html(lpd);
 
    DDHtmlMaDetails mad("ma","Materials");
    dd_to_html(mad);
 
    DDHtmlSoDetails sod("so","Solids");
    dd_to_html(sod);

    DDHtmlRoDetails rod("ro","Rotations");
    dd_to_html(rod);

    DDHtmlSpDetails spd("sp","Specifics (SpecPars)");
    dd_to_html(spd);

    std::ofstream fr;

    fr.open("index.html");
    dd_html_menu_frameset(fr);
    fr.close();

    fr.open("menu.html");
    dd_html_menu(fr);
    fr.close();

    fr.open("lp/index.html");
    dd_html_frameset(fr);
    fr.close();
  
    fr.open("ma/index.html");
    dd_html_frameset(fr);
    fr.close();

    fr.open("so/index.html");
    dd_html_frameset(fr);
    fr.close();

    fr.open("ro/index.html");
    dd_html_frameset(fr);
    fr.close();

    fr.open("sp/index.html");
    dd_html_frameset(fr);
    fr.close();

    return 0;
  
  }
  catch (cms::Exception& e) // DDD-Exceptions are simple string for the Prototype
    {
      std::cerr << "DDD-PROBLEM:" << std::endl 
		<< e << std::endl;
    }  

}
