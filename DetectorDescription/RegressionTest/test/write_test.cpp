#include <iostream>
#include <fstream>
#include <cstdlib>
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
#include "DetectorDescription/Core/interface/DDStreamer.h"
#include "DetectorDescription/RegressionTest/interface/DDExpandedViewDump.h"


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
  size_t skip = 777;
  size_t mx(0);
  if (getenv("DDSKIP")) {
    skip = atoi(getenv("DDSKIP"));
  }
  if (getenv("DDMAX")) {
    mx = atoi(getenv("DDMAX"));
  }
  cout << "DDSKIP  =" << skip << endl;
  if (getenv("DDEXDUMP"))
    cout << "DDEXDUMP=" << getenv("DDEXDUMP") << endl;
 
  string configfile="configuration.xml";
  string persoutput="pers.txt";
  if (getenv("DDPERSOUTPUT")) {
    persoutput = getenv("DDPERSOUTPUT");
    cout << "DDPERSOUTPUT=" << persoutput << endl;
  }
  else {
    cout << "DDPERSOUTPUT not set, using " << persoutput << endl;
  }
  if (getenv("DDCONFIG")) {
    configfile = getenv("DDCONFIG");
    cout << "DDCONFIG=" << configfile << endl;
  }
  else {
    cout << "DDCONFIG not set, using " << configfile << endl;
  }

 
  /*
  cout << "sizeof(DDValue)=" << sizeof(DDValue) << endl;
  vector<DDValue> vec_val;
  DDValue v1("Liendl","martin");
  DDValue v2("Liendl","arno");
  vec_val.push_back(v2);
  vec_val.push_back(v1);
  cout << v1.id() << ' ' << v2.id() << endl;
  cout << vec_val[0] << ' ' << vec_val[1] << endl;
  */
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
  myP->SetConfig(configfile);

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
  DDStreamer streamer;
  //ofstream file("pers.txt",ios_base::binary);
  ofstream file(persoutput.c_str());

  cout << "STREAMING:" << endl;
  streamer.write(file);  
  
  DDName n("GLOBAL","GLOBAL");
  cout << "name.id(): " << n.id()<< ' ' << endl;
  if(getenv("DDEXDUMP")){
    ofstream exd_file("exdump.txt");
    DDCompactView cppv;
    DDExpandedView exv(cppv);
    DDExpandedViewDump(exd_file,exv,skip,mx);
  }
  TimingReport* tr = TimingReport::current();
  tr->dump(cout);    
  return 0;
  
}
catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
{
   cerr << "DDD-PROBLEM:" << endl 
        << e << endl;
   throw;
}  

}
