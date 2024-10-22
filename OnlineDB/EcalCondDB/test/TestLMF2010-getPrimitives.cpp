#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/LMFDefFabric.h"
#include "OnlineDB/EcalCondDB/interface/LMFLaserPulseDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFPnPrimDat.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"
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
  CondDBApp(string sid, string user, string pass) {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface(sid, user, pass);
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() { delete econn; }

  void doRead() {
    LMFPnPrimDat pnPrim(econn);
    EcalLogicID ecid = econn->getEcalLogicID(1131190000);
    Tm t;
    t.setToString("2010-09-21 00:00:00");
    pnPrim.fetch(ecid, t);
    pnPrim.setMaxDataToDump(10);
    pnPrim.dump();
    cout << "These data were taken on " << pnPrim.getSubrunStart().str() << endl;
    LMFPnPrimDat *p = new LMFPnPrimDat;
    LMFPnPrimDat *n = new LMFPnPrimDat;
    pnPrim.getPrevious(p);
    pnPrim.getNext(n);
    p->dump();
    n->dump();
    delete p;
    delete n;
  }

private:
  CondDBApp() = delete;  // hidden default constructor
  EcalCondDBInterface *econn;
};

int main(int argc, char *argv[]) {
  string sid;
  string user;
  string pass;

  if (argc != 4) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];

  try {
    CondDBApp app(sid, user, pass);
    app.doRead();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
