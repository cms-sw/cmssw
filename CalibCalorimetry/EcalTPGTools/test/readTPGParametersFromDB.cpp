#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGDBApp.h"

#include <cassert>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include <fstream>
#include <iostream>



int main (int argc, char* argv[])
{
  string host;
  string sid;
  string user;
  string pass;
  string sport;
  string smin_run;
  string sn_run;

  if (argc != 8) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <port> <min_run> <n_run>" << endl;
    exit(-1);
  }


  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  sport = argv[5];
  int port=atoi(sport.c_str());
  smin_run = argv[6];
  int min_run=atoi(smin_run.c_str());
  sn_run = argv[7];
  int n_run=atoi(sn_run.c_str());

  try {
    EcalTPGDBApp app( sid, user, pass);

    int i ; 
    //app.readTPGPedestals(i);
    //app.writeTPGLUT();
    //app.writeTPGWeights();

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
