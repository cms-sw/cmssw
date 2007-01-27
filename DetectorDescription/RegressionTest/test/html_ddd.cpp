#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
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

#include "CARF/DDDInterface/interface/GeometryConfiguration.h"

//#include "DetectorDescription/Core/interface/graph_path.h"
//typedef GraphPath<DDLogicalPart,DDPosData*> GPathType;

// The DDD user-code after XML-parsing is located
// in DetectorDescription/Core/src/tutorial.cc
// Please have a look to all the commentary therein.
#include "DetectorDescription/Core/src/tutorial.h"

// html generator
#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"

#include "Utilities/Notification/interface/TimingReport.h"
#include "Utilities/Notification/interface/TimerProxy.h"

using namespace std;
int main(int argc, char *argv[])
{
  static TimerProxy timer_("main()");
  TimeMe t(timer_,false);
 
try { // DDD Prototype can throw DDException defined in DetectorDescription/Core/interface/DDException.h
  
  // Initialize a DDL Schema aware parser for DDL-documents
  // (DDL ... Detector Description Language)
  cout << "initialize DDL parser" << endl;
  DDLParser* myP = DDLParser::Instance();

  cout << "about to set configuration" << endl;
  string configfile("configuration.xml");
  if (argc==2) {
    configfile = argv[1];
  }
  GeometryConfiguration documentProvider(configfile);
  cout << "about to start parsing" << endl;
  int parserResult = myP->parse(documentProvider);

  if (parserResult != 0) {
    cout << " problem encountered during parsing. exiting ... " << endl;
    exit(1);
  }
  cout << " parsing completed" << endl;
  
  cout << endl << endl << "Start checking!" << endl << endl;
  
  //DDCheckMaterials(cout);
  DDCheck(cout);
  
  /* Now start the 'user-code' */
  
  typedef DDHtmlFormatter::ns_type ns_type;
  
  ns_type names;
  { 
    static TimerProxy timer2_("findNameSpaces<LogicalPart>(..)");
    TimeMe t(timer2_,false);
    DDLogicalPart lp;
    findNameSpaces(lp, names);
  }
  
  cout << names.size() << " namespaces found: " << endl << endl;
  ns_type::const_iterator nit = names.begin();
  for(; nit != names.end(); ++nit) {
    cout << nit->first 
         << " has " 
         << nit->second.size()
         << " entries." << endl; 
  }
 
  DDErrorDetection ed;
  ed.scan();
  ed.report(cout);


  
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

  ofstream fr;

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
  

 
  TimingReport* tr = TimingReport::current();
  tr->dump(cout);    
  return 0;
  
}
catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
{
   cerr << "DDD-PROBLEM:" << endl 
        << e << endl;
}  

}
