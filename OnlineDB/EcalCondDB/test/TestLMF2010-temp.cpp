#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include <limits.h>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"

using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string sid, string user, string pass, run_t r)
  {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface( sid, user, pass );
      run = r;
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() 
  {
    delete econn;
  }

  void doRun() {
    LocationDef my_locdef;
    my_locdef.setLocation( "P5_Co" );
    
    RunTypeDef my_rundef;
    my_rundef.setRunType( "PHYSICS" );
    
    RunTag  my_runtag;
    my_runtag.setLocationDef( my_locdef );
    my_runtag.setRunTypeDef(  my_rundef );
    my_runtag.setGeneralTag( "GLOBAL" );
    
    // Look for new runs in the DB
    RunList runlist = econn->fetchRunList( my_runtag, 161900, 161976 );
    std::vector<RunIOV> run_vec =  runlist.getRuns();

    std::cout << "ALL: " << run_vec.size() << std::endl;
    for (unsigned int i = 0; i < run_vec.size(); i++) {
      std::cout << i << " " << run_vec[i].getRunNumber() << std::endl;
    }

    run_vec.clear();
    runlist = econn->fetchNonEmptyRunList( my_runtag, 161900, 161976 );
    run_vec =  runlist.getRuns();
    
    std::cout << "NON EMPTY: " << run_vec.size() << std::endl;  
    for (unsigned int i = 0; i < run_vec.size(); i++) {
      std::cout << i << " " << run_vec[i].getRunNumber() << std::endl;
    }
    runlist = econn->fetchNonEmptyGlobalRunList( my_runtag, 161900, 161976 );
    run_vec.clear();
    run_vec =  runlist.getRuns();
    
    std::cout << "NON EMPTY GLOBAL: " << run_vec.size() << std::endl;  
    for (unsigned int i = 0; i < run_vec.size(); i++) {
      std::cout << i << " " << run_vec[i].getRunNumber() << std::endl;
    }
  }

private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  run_t run;
};

int main (int argc, char* argv[])
{
  string sid;
  string user;
  string pass;

  if (argc != 5) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <run>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  int run = atoi(argv[4]);

  try {
    CondDBApp app(sid, user, pass, run);
    app.doRun();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
