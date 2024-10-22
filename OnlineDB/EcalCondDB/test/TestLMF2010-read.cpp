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

  void doRead() {
    std::string location = "P5_Co";
    RunIOV runiov = econn->fetchRunIOV(location, run);
    cout << "Attaching data to Run " << runiov.getRunNumber() << endl;
    // now create a sequence
    LMFSeqDat seq(econn);
    seq.setRunIOV(runiov).setSequenceNumber(1);
    //    seq.debug();
    seq.fetchID();
    //    seq.dump();
    LMFRunIOV lmfruniov(econn);
    int type = 1;
    int color = 1;
    for (int lmr = 1; lmr < 93; lmr++) {
      bool res = econn->fetchLMFRunIOV(seq, lmfruniov, lmr, type, color);
      std::cout << "LMR: " << lmr << " - " << res << std::endl;
      if ((lmr == 1) || (lmr == 92)) {
        lmfruniov.dump();
      }
    }
    /*
    lmfruniov.debug();
    cout << econn->getEnv() << " " << econn->getConn() << endl;
    std::list< LMFRunIOV > iov_l = lmfruniov.fetchBySequence( seq );
    cout << iov_l.size() << endl;
    exit(0);
    lmfruniov.setSequence(seq).setLmr(3);
    lmfruniov.fetchID();
    lmfruniov.dump();
    vector<LMFDat*> v; 
    LMFRunDat *lmfrundat = new LMFRunDat(econn);
    LMFTestPulseConfigDat *lmfconfigdat = new LMFTestPulseConfigDat(econn);
    LMFLaserConfigDat *lmflaserconfdat = new LMFLaserConfigDat(econn);
    LMFLaserPulseDat *lmfbluepulsedat = new LMFLaserPulseDat("BLUE");
    lmfbluepulsedat->setConnection(econn->getEnv(), econn->getConn());
    v.push_back(lmfrundat);
    v.push_back(lmfconfigdat);
    v.push_back(lmflaserconfdat);
    v.push_back(lmfbluepulsedat);
    for (unsigned int i = 0; i < v.size(); i++) {
      v[i]->setLMFRunIOV(lmfruniov);
      v[i]->fetch();
      v[i]->setMaxDataToDump(10);
      v[i]->dump();
    }
    delete lmfrundat;
    delete lmfconfigdat;
    delete lmflaserconfdat;
    delete lmfbluepulsedat;
    */
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
    app.doRead();
  } catch (exception& e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
