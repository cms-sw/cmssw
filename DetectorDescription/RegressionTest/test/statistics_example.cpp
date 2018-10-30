#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/RegressionTest/src/ddstats.h"

using namespace std;
int main(int argc, char *argv[])
{
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
  if (argc==1) {
  myP->SetConfig("configuration.xml");
  } else
    myP->SetConfig(*(argv+1));
 
  //LINUX ONLY!!!!
  //ifstream proc("/proc/self/status") ;
  //cout << proc << endl;
  //proc.close();
  string nix;
  cout << "about to start parsing" << endl;
  cout << " check process size now, and press \'c\'+<return> to continue .." << endl;
  cin >> nix;
  int parserResult = myP->StartParsing();
  if (parserResult != 0) {
    cout << " problem encountered during parsing. exiting ... " << endl;
    exit(1);
  }
  cout << " parsing completed" << endl;
  cout << " check process size now, and press \'c\'+<return> to continue .." << endl;
  cin >> nix;
  
  cout << endl << endl << "Start checking!" << endl << endl;
  
  /* Currently only materials are checked.
     (Checking means that additional consitency test are done which
      can not be guaranteed to be ok by simple Schema conformance)
      Functional SW will automatically call various Tests after parsing 
      is finished)
  */    


  //DDCheckMaterials(cout);
  DDCheck(cout);

  //Statistics
  ddstats(cout);
  cout << " check process size now, and press \'c\'+<return> to continue .." << endl;
  cin >> nix;
  cout << "Clearing CompactView ..." << endl;
  DDCompactView cpv;
  cpv.clear();
  cout << " check process size now, and press \'c\'+<return> to continue .." << endl;
  cin >> nix;
    
  return 0;
  
}
catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
{
   cerr << "DDD-PROBLEM:" << endl 
        << e << endl;
}  

}
