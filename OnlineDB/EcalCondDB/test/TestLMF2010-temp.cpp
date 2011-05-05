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
    std::cout << econn->getDetIdFromLogicId(200) << std::endl;
    int ebdetid = econn->getDetIdFromLogicId(1011350303);
    EBDetId e(ebdetid);
    std::cout << econn->getDetIdFromLogicId(1011350303) << std::endl;
    std::cout << e.ieta() << " " << e.iphi() << std::endl;
    int eedetid = econn->getDetIdFromLogicId(2010015016);
    std::cout << eedetid << std::endl;
    EEDetId ee(eedetid);
    std::cout << ee.zside() << " " << ee.ix() << " " << ee.iy() << std::endl;
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
