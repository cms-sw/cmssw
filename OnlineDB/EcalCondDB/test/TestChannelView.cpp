#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_dcu_types.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp( string sid, string user, string pass)
  {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface( sid, user, pass );
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }

    locations[0] = "H4";
    locations[1] = "867-1";
    locations[2] = "867-2";
    locations[3] = "867-3";
  }



  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() 
  {
    delete econn;
  }


  /**
   * test read 
   */
  void test()  {
    cout << "test of the channelview table..." << endl;
    

    
    // Get channel ID for SM 10, crystal c      
    // int c = 1;
    //   EcalLogicID ecid;
    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSet("EB_elec_crystal_number", 10, 10, 1, 10,EcalLogicID::NULLID, EcalLogicID::NULLID,"EB_crystal_number");
    //    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);
    for (int i=0; i<(int)ecid_vec.size() ; i++){
      int id1=ecid_vec[i].getID1();
      int id2=ecid_vec[i].getID2();
      int log_id=ecid_vec[i].getLogicID();
      cout << "id1="<<id1<<" id2="<<id2<<" logic_id="<<log_id<<endl;
    }

  };
 




private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  string locations[4];
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  int tablesTried;
  int tablesOK;

 
  /**
   *   Print out a DCUTag
   */
} ;



int main (int argc, char* argv[])
{
  string host;
  string sid;
  string user;
  string pass;

  if (argc != 5) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];

  try {
    CondDBApp app(sid, user, pass);

    app.test();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
