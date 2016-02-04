#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGDBApp.h"

#include <cassert>
#include <string>
#include <fstream>
#include <iostream>



int main (int argc, char* argv[])
{
  std::string host;
  std::string sid;
  std::string user;
  std::string pass;
  std::string sport;
  std::string smin_run;
  std::string sn_run;

  if (argc != 8) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv[0] << " <host> <SID> <user> <pass> <port> <min_run> <n_run>" << std::endl;
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
    std::cout << "ERROR:  " << e.what() << std::endl;
  } catch (...) {
    std::cout << "Unknown error caught" << std::endl;
  }

  std::cout << "All Done." << std::endl;

  return 0;
}
