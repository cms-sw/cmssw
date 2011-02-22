#include <iostream>
#include <fstream>

#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"

// required for main() in cmssw framework
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "DetectorDescription/Core/src/LogicalPart.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"

int main(int argc, char *argv[])
{
  // required for main() in cmssw framework
  std::string const kProgramName = argv[0];
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
    std::string configfile("DetectorDescription/RegressionTest/test/dddhtml/");
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
  catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
    {
      std::cerr << "DDD-PROBLEM:" << std::endl 
		<< e << std::endl;
    }  

}
