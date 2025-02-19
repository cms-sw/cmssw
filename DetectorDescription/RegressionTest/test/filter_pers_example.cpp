#include <iostream>
#include <fstream>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDStreamer.h"
// The DDD user-code after XML-parsing is located
// in DetectorDescription/Core/src/tutorial.cc
// Please have a look to all the commentary therein.
#include "DetectorDescription/Core/src/tutorial.h"

using namespace std;
int main(int argc, char *argv[])
{
   static TimerProxy timer_("main()");
   TimeMe t(timer_,false);
   string persinput;
  if (getenv("DDPERSINPUT")) {
    persinput = getenv("DDPERSINPUT");
    cout << "DDPERSINPUT=" << persinput << endl;
  }
  else {
    cout << "DDPERSINPUT not set, using " << persinput << endl;
  }

  //DDAlgoInit();
try { // DDD Prototype can throw DDException defined in DetectorDescription/Core/interface/DDException.h
  
  // Initialize a DDL Schema aware parser for DDL-documents
  // (DDL ... Detector Description Language)
  cout << "using DDStreamer and ./pers.txt for reading stuff in ..." << endl;
  DDStreamer streamer;
  ifstream file(persinput.c_str());
  if(!file) throw DDException("Could not open file=" + persinput);
  streamer.read(file);  
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
  DDSpecificsFilter specfilter;
  DDValue filterval("MuStructure",1);
  specfilter.setCriteria(filterval,  // name and value to be compared
                          DDSpecificsFilter::equals, // comparison operation
			  DDSpecificsFilter::AND, // always AND for the first of the criteria
			  false); // do not compare a string-value, but a double!
  
  DDCompactView cv;
  DDFilteredView fv(cv);
  fv.addFilter(specfilter, DDFilteredView::AND); 			  
  
  // try to extract the specifics
  
  DDValue valUpar("upar");
  DDValue valMuStruct("MuStructure");
  cout <<"Listing the first 10 specifics named 'upar':" << endl;
  int i=0;
  while (fv.next() && (i<10)) {
    DDsvalues_type sv(fv.mergedSpecifics());
    cout << "logicalpart=" << fv.logicalPart().name()
         << ": ";  
    if (DDfetch(&sv,valUpar)) {
      cout << valUpar << " " << endl;
      cout << " hierarchy:" << fv.geoHistory() << endl;
      //cout << " eval-status: " << valUpar.vecPair_->first << endl;
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
      cout << valMuStruct << endl;
    }
  
    cout << " position: trans=" << fv.translation() << "  rot.ax=" << fv.rotation().axis() 
         << "  rot.angle=" << fv.rotation().delta() << endl;
    cout << endl; 
    ++i;
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
