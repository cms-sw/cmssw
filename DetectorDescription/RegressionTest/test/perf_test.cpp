#include <iostream>

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
#include "DetectorDescription/Core/interface/DDValue.h"

//#include "DetectorDescription/Core/interface/graph_path.h"
//typedef GraphPath<DDLogicalPart,DDPosData*> GPathType;

// The DDD user-code after XML-parsing is located
// in DetectorDescription/Core/src/tutorial.cc
// Please have a look to all the commentary therein.
#include "DetectorDescription/Core/src/tutorial.h"

using namespace std;
int main(int argc, char *argv[])
{
  static TimerProxy timer_("main()");
  TimeMe t(timer_,false);
 
  cout << "sizeof(DDValue)=" << sizeof(DDValue) << endl;
  vector<DDValue> vec_val;
  DDValue v1("Liendl","martin");
  DDValue v2("Liendl","arno");
  vec_val.push_back(v2);
  vec_val.push_back(v1);
  cout << v1.id() << ' ' << v2.id() << endl;
  cout << vec_val[0] << ' ' << vec_val[1] << endl;
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
  DDSolid::iterator<DDSolid> it, ed;
  ed.end();
  cout << "Listing the first 5 solids:" << endl;
  for (int j=0; it != ed && j<5; ++j, ++it) {
    cout << *it << endl;
  }

 
  DDCompactView cv;
  
  
  DDSpecificsFilter specfilter_1;
  //DDValue filterval("MuStructure","xyz");
  DDValue filterval_1("CopyNoTag",000);
  specfilter_1.setCriteria(filterval_1,  // name and value to be compared
                       DDSpecificsFilter::bigger, // comparison operation
			  DDSpecificsFilter::AND, // always AND for the first of the criteria
			  false, // as string ?
			  true); // merged specs ?
  
  DDSpecificsFilter specfilter_2;			  
  DDValue filterval_2("MuStructure","xyz");
  specfilter_2.setCriteria(filterval_2,  // name and value to be compared
                       DDSpecificsFilter::not_equals, // comparison operation
			  DDSpecificsFilter::AND, // always AND for the first of the criteria
			  true, // as string ?
			  true); // merged specs ?
			  
  DDFilteredView fv(cv);
  fv.addFilter(specfilter_1, DDFilteredView::AND); 			  
  //fv.addFilter(specfilter_2, DDFilteredView::AND);
  // try to extract the specifics
  
  DDValue valUpar("upar");
  DDValue valMuStruct("MuStructure");
  cout <<"Listing the first 10 specifics named 'upar':" << endl;
  
  {
  static TimerProxy timer2_("loop over FilteredView");
  TimeMe t(timer2_,false);

  int i=0;
  while (fv.next()) {

 /* 
    DDsvalues_type sv(fv.mergedSpecifics());
    cout << "logicalpart=" << fv.logicalPart().name()
         << ": ";  
    if (DDfetch(&sv,valUpar)) {
      cout << valUpar << " " << endl;
      cout << " hierarchy:" << fv.geoHistory() << endl;
 
      cout << " valUpar is a vector of doubles, if you want it to be one:" << endl;
      const vector<double> & dvec = valUpar.doubles();
      cout << "  dvec.size()=" << dvec.size() << endl;
      cout << "  dvec[2]=" << dvec[2] << endl;
      
      cout << " valUpar is a vector of strings, if you want it to be one:" << endl;
      const vector<string> & svec = valUpar.strings();
      cout << "  svec.size()=" << svec.size() << endl;
      cout << "  svec.[2]=" << svec[2] << endl;
    }
    if (DDfetch(&sv,valMuStruct)) {
      cout << valMuStruct;
    }  
    cout << endl << endl; 
*/    
    ++i;
  }
  cout << "The FilteredView has " << i << " nodes." << endl; 
  }
  {
    static TimerProxy timer3_("loop over ExpandedView");
    TimeMe t(timer3_,false);
    int i=0;
    DDExpandedView ex(cv);
    while (ex.next()) ++i;
    cout << "The ExpandedView has " << i << " nodes." << endl;
  }
  
  // inverting the graph
  graph_type invers;
  {   
    static TimerProxy timer4_("graph::invert(..)");
    TimeMe t(timer4_,false);
  
    const graph_type & gr = DDCompactView().graph();
    gr.invert(invers);
  }
  
  // finding the roots of the inverted-graph
  graph_type::edge_list el;
  {
    static TimerProxy timer5_("graph::roots(..)");
    TimeMe t(timer5_,false);
    
    invers.findRoots(el);
  }
  
  graph_type::edge_list::iterator elit = el.begin();
  for (; elit != el.end(); ++elit) {
    cout << invers.nodeData(elit->first).name() << ' ';
  }
  cout << endl << "there were " << el.size() << " roots in the inverted graph" << endl;
  cout << endl;
  
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
