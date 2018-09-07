#include <iostream>
#include <cstdlib>
#include <vector>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/src/tutorial.h"
#include "DetectorDescription/Core/src/Specific.h"
#include "DetectorDescription/Core/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDComparator.h"

/**********

SEE LINES 93ff

***********/
using namespace DDI;
using namespace std;

// Helper class
class PSCreator : public Specific
{
public:
  PSCreator(const vector<string> & s, DDsvalues_type dummy) 
  : Specific(s, dummy ) 
  { }
  vector<DDPartSelection> selections() { return Specific::partSelections_; } 
};



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
  myP->SetConfig("configuration.xml");

  cout << "about to start parsing" << endl;
  int parserResult = myP->StartParsing();
  if (parserResult != 0) {
    cout << " problem encountered during parsing. exiting ... " << endl;
    exit(1);
  }
  cout << " parsing completed" << endl;
  
  cout << endl << endl << "Start checking!" << endl << endl;
  
  /* Currently only materials are checked.
     (Checking means that additional consitency test are done which
      can not be guaranteed to be ok by simple Schema conformance)
      Functional SW will automatically call various Tests after parsing 
      is finished)
  */    


  //DDCheckMaterials(cout);
  DDCheck(cout);
  
  /* Now start the 'user-code' */
  bool loop=true;
 
 DDCompactView cpv; DDExpandedView ex(cpv);  
 while(loop) {
    vector<string> selV(1);
    cout << "Part-Selection (//doc1:name1//doc2:name2[cpno2]/nam4 ...)" << endl;
    cin >> selV[0];
    if (selV[0]=="end") loop=false;
    DDsvalues_type dummy;
    vector<DDPartSelection> partSelV = PSCreator(selV,dummy).selections();
    int I = partSelV.size();int i=0;
    for (; i<I; ++i) {
       bool go=true;
       while(go) {
          if (DDCompareEqual(ex.geoHistory(),partSelV[i])(ex.geoHistory(),partSelV[i])) {
	    // HERE THE SELECTION STRING MATCHES A PART IN EXPANDEDVIEW:
	    cout << ex.geoHistory() << endl;
	    cout << ex.translation() << endl;
            // << ' ' << ex.rotation() << endl;
	  }
	  go=ex.next();
       }
       ex.reset();
    }
    
    
 }     
  return 0;
  
}
catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
{
   cerr << "DDD-PROBLEM:" << endl 
        << e << endl;
}  

}
