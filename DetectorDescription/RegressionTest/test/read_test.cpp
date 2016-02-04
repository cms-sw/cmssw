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
  size_t skip(777);
  size_t mx(0);
  if (getenv("DDSKIP")) {
    skip = atoi(getenv("DDSKIP"));
  } 
  if (getenv("DDMAX")) {
    mx = atoi(getenv("DDMAX"));
  }
  std::cout << "DDSKIP  =" << skip << endl;
  if (getenv("DDEXDUMP")) {
   std::cout << "DDEXDUMP=" << getenv("DDEXDUMP") << endl << flush;  
  }

  string persinput="pers.txt";
  string persoutput="pers2.txt";
  string configfile="";
  if (getenv("DDPERSINPUT")) {
    persinput = getenv("DDPERSINPUT");
    cout << "DDPERSINPUT=" << persinput << endl;
  }
  else {
    cout << "DDPERSINPUT not set, using " << persinput << endl;
  }
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

  try { 
 
    ifstream file(persinput.c_str());
    ofstream outf(persoutput.c_str());
    if (!file) throw DDException("Error, can't open persistency file " + persinput); 
    if (!outf) throw DDException("Error, can't open output file " + persoutput); 
   DDStreamer streamer;
  

  std::cout << "READING:" << std::endl << std::flush;  
  streamer.read(file);
  if (configfile!="") {
    cout << "initialize DDL parser" << endl;
    DDLParser* myP = DDLParser::Instance();
    cout << "about to set configuration" << endl;
    myP->SetConfig(configfile);
    cout << "about to start parsing" << endl;
    int parserResult = myP->StartParsing();
    if (parserResult != 0) {
      cout << " problem encountered during parsing. exiting ... " << endl;
      throw DDException("XML parsing failed!");
    }
    cout << " parsing completed" << endl;
  }
  cout << "READING DONE, WRITING AGAIN" << endl;
  //ofstream outf("pers2.txt",ios_base::binary);
  streamer.write(outf);
  cout << "DONE" << endl;  
     
  DDName n("GLOBAL","GLOBAL");
  cout << "name.id(): " << n.id() << endl;
  if(getenv("DDEXDUMP")){
    ofstream exd_file("exdump2.txt");
    DDCompactView cpv;
    DDExpandedView exv(cpv);
    DDExpandedViewDump(exd_file,exv,skip,mx);
  }
  cout << "Deleting CompactView..." << endl;
  
}
catch (DDException& e) // DDD-Exceptions are simple string for the Prototype
{
   cerr << "DDD-PROBLEM:" << endl 
        << e << endl;
}  

  TimingReport* tr = TimingReport::current();
  tr->dump(cout);    
  DDCompactView delme;
  delme.clear();
 
return 0;
}
