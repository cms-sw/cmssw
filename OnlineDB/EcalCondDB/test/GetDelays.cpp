#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include <climits>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class CondDBApp {
public:
  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string sid, string user, string pass, run_t r) {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface(sid, user, pass);
      run = r;
      cout << "Done." << endl;
    } catch (runtime_error& e) {
      cerr << e.what() << endl;
      exit(-1);
    }
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() { delete econn; }

  void doRun() {
    RunIOV iov = econn->fetchRunIOV("P5_Co", run);
    std::list<ODDelaysDat> delays = econn->fetchFEDelaysForRun(&iov);
    std::list<ODDelaysDat>::const_iterator i = delays.begin();
    std::list<ODDelaysDat>::const_iterator e = delays.end();
    while (i != e) {
      std::cout << "SM: " << i->getSMId() << " FED: " << i->getFedId() << " Delay: " << i->getTimeOffset() << std::endl;
      i++;
    }
  }

private:
  CondDBApp() = delete;  // hidden default constructor
  EcalCondDBInterface* econn;
  run_t run;
};

int main(int argc, char* argv[]) {
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
  } catch (exception& e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
