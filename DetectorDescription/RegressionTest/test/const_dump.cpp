#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDD.h"

// DDD Interface in CARF
#include "CARF/DDDInterface/interface/GeometryConfiguration.h"

// Error Detection
#include "DetectorDescription/RegressionTest/interface/DDHtmlFormatter.h"

//#include "DetectorDescription/Core/interface/graph_path.h"
//typedef GraphPath<DDLogicalPart,DDPosData*> GPathType;

// The DDD user-code after XML-parsing is located
// in DetectorDescription/Core/src/tutorial.cc
// Please have a look to all the commentary therein.

#include "Utilities/Notification/interface/TimingReport.h"
#include "Utilities/Notification/interface/TimerProxy.h"

using namespace std;
namespace DD { } using namespace DD;

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
  /* The configuration file tells the parser what to parse.
     The sequence of files to be parsed does not matter but for one exception:
     XML containing SpecPar-tags must be parsed AFTER all corresponding
     PosPart-tags were parsed. (Simply put all SpecPars-tags into seperate
     files and mention them at end of configuration.xml. Functional SW 
    will not suffer from this restriction).
  */  
  //myP->SetConfig("configuration.xml");
  cout << "about to start parsing" << endl;
  string configfile("configuration.xml");
  if (argc==2) {
    configfile = argv[1];
  }
  GeometryConfiguration documentProvider("configuration.xml");
  int parserResult = myP->parse(documentProvider);
  if (parserResult != 0) {
    cout << " problem encountered during parsing. exiting ... " << endl;
    exit(1);
  }
  cout << " parsing completed" << endl;
  
  cout << endl << endl << "Start checking!" << endl << endl;
 
  DDErrorDetection ed;
  ed.scan();
  ed.report(cout);

  Constant::iterator<Constant> cit(Constant::begin()), ced(Constant::end());
  for(; cit != ced; ++cit) {
    cout << *cit << endl;
  }
  Vector::iterator<Vector> vit(Vector::begin()), ved(Vector::end());
  for(; vit != ved; ++vit) {
    cout << *vit << endl;
  }


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
