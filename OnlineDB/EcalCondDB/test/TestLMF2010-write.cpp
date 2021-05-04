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
    std::string location = "P5_Co";
    RunIOV runiov = econn->fetchRunIOV(location, run);
    cout << "Attaching data to Run " << runiov.getRunNumber() << endl;
    // now create a sequence
    LMFSeqDat seq(econn);
    seq.setRunIOV(runiov).setSequenceNumber(1);
    seq.debug();
    seq.fetchID();
    seq.dump();
    LMFRunIOV lmfruniov(econn);
    lmfruniov.debug();
    cout << econn->getEnv() << " " << econn->getConn() << endl;
    std::list<LMFRunIOV> iov_l = lmfruniov.fetchBySequence(seq);
    cout << iov_l.size() << endl;
    exit(0);
    lmfruniov.setSequence(seq).setLmr(3);
    lmfruniov.fetchID();
    lmfruniov.dump();
    vector<LMFDat *> v;
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
  }

  void doRun() {
    LMFDefFabric fabric(econn);
    // find the RunIOV corresponding to a given run
    std::string location = "H4B";
    RunIOV runiov = econn->fetchRunIOV(location, run);
    cout << "Attaching data to Run " << runiov.getRunNumber() << endl;
    // now create a sequence
    LMFSeqDat seq;
    seq.setRunIOV(runiov).setSequenceNumber(1);
    Tm start;
    start.setToCurrentLocalTime();
    seq.setSequenceStart(start);
    seq.setVersions(1, 1);
    seq.dump();
    econn->insertLmfSeq(&seq);
    // note that after insert the ID is different from zero
    seq.dump();
    // a second insertion will not cause any write
    econn->insertLmfSeq(&seq);
    seq.dump();
    // change a parameter (i.e. the seq_num)
    seq.setSequenceNumber(2);
    // the ID has changed
    seq.dump();
    // now write and check that we have a different ID
    econn->insertLmfSeq(&seq);
    seq.dump();
    // now create the corresponding LMF_RUN_IOV
    LMFRunIOV lmfruniov(econn->getEnv(), econn->getConn());
    // we need to associate a tag to the LMF_RUN_IOV. For all classes we have
    // different constructors.
    LMFRunTag lmfruntag(econn);
    lmfruntag.setGeneralTag("gen").setVersion(2);  // operators can be catenated
    if (lmfruntag.exists()) {
      lmfruntag.dump();
      cout << "OK" << endl;
    } else {
      lmfruntag.dump();
      cout << "Does not exists" << endl;
    }
    // we can just get the tags from the DB
    auto listOfTags = lmfruntag.fetchAll();
    cout << "Found " << listOfTags.size() << " tags" << endl;
    for (auto &tag : listOfTags) {
      tag->dump();
    }
    // we can also get the tags from the fabric
    lmfruntag = fabric.getRunTag("gen", 3);
    lmfruntag.dump();
    // other possibilities
    lmfruntag = fabric.getRunTags().front();
    lmfruntag.dump();
    lmfruntag = fabric.getRunTags().back();
    lmfruntag.dump();
    // assign this tag to the run iov
    lmfruniov.setLMFRunTag(lmfruntag);
    lmfruniov.setSequence(seq);
    lmfruniov.setLmr(1);
    lmfruniov.setColor("blue");
    lmfruniov.setTriggerType("las");
    lmfruniov.dump();
    cout << "Starting filling primitives" << endl;
    // get the list of logic_ids
    vector<EcalLogicID> ecid_vec;
    int sm_min = 1;
    int sm_max = 36;
    int ch_min = 0;
    int ch_max = 1;
    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_LM_side",
                                               sm_min,
                                               sm_max,
                                               ch_min,
                                               ch_max,
                                               EcalLogicID::NULLID,
                                               EcalLogicID::NULLID,
                                               "EB_crystal_number",
                                               1234);
    cout << ecid_vec.size() << endl;
    // create random data
    vector<EcalLogicID>::const_iterator i = ecid_vec.begin();
    vector<EcalLogicID>::const_iterator e = ecid_vec.end();
    LMFRunDat rundat(econn);
    rundat.setLMFRunIOV(lmfruniov);
    rundat.dump();
    lmfruniov.dump();
    cout << "Filling with data" << endl;
    while (i != e) {
      EcalLogicID logic_id = *i;
      int nevts = rand();
      int quality = rand() % 10;
      rundat.setData(logic_id, nevts, quality);
      cout << i->getLogicID() << '\r';
      i++;
    }
    cout << endl;
    econn->insertLmfDat(&rundat);
    cout << "LMFRunDAT written" << endl;
    // pulse + laser config
    LMFTestPulseConfigDat tpDat(econn);
    tpDat.setLMFRunIOV(lmfruniov);
    LMFLaserConfigDat lcDat(econn);
    lcDat.setLMFRunIOV(lmfruniov);
    i = ecid_vec.begin();
    int gain[3] = {1, 6, 12};
    while (i != e) {
      EcalLogicID logic_id = *i;
      i++;
      int g = rand() % 3;
      tpDat.setData(logic_id, gain[g], rand(), rand(), rand());
      vector<float> random_data;
      for (int k = 0; k < 8; k++) {
        random_data.push_back((float)rand() / static_cast<float>(RAND_MAX));
      }
      lcDat.setData(logic_id, random_data);
    }
    econn->insertLmfDat(&tpDat);
    econn->insertLmfDat(&lcDat);
    // test primitives
    cout << "Testing primitives" << endl;
    LMFLaserPulseDat laserbluepulsedat(econn, "BLUE");
    LMFLaserPulseDat laserirpulsedat(econn, "IR");
    LMFPnPrimDat pnPrimDat(econn, "BLUE", "LASER");
    laserbluepulsedat.setLMFRunIOV(lmfruniov);
    laserirpulsedat.setLMFRunIOV(lmfruniov);
    pnPrimDat.setLMFRunIOV(lmfruniov);
    i = ecid_vec.begin();
    while (i != e) {
      EcalLogicID id = *i;
      laserbluepulsedat.setFitMethod(id, rand());
      laserbluepulsedat.setMTQAmplification(id, rand());
      laserbluepulsedat.setMTQTime(id, rand());
      laserbluepulsedat.setMTQRise(id, rand());
      laserbluepulsedat.setMTQFWHM(id, rand());
      laserbluepulsedat.setMTQFW20(id, rand());
      laserbluepulsedat.setMTQFW80(id, rand());
      laserbluepulsedat.setMTQSliding(id, rand());
      laserbluepulsedat.setVersions(id, 1, 1);

      laserirpulsedat.setFitMethod(id, rand());
      laserirpulsedat.setMTQAmplification(id, rand());
      laserirpulsedat.setMTQTime(id, rand());
      laserirpulsedat.setMTQRise(id, rand());
      laserirpulsedat.setMTQFWHM(id, rand());
      laserirpulsedat.setMTQFW20(id, rand());
      laserirpulsedat.setMTQFW80(id, rand());
      laserirpulsedat.setMTQSliding(id, rand());
      laserirpulsedat.setVersions(id, 1, 1);

      pnPrimDat.setMean(id, rand());
      pnPrimDat.setRMS(id, rand());
      pnPrimDat.setM3(id, rand());
      pnPrimDat.setPNAoverBMean(id, rand());
      pnPrimDat.setPNAoverBRMS(id, rand());
      pnPrimDat.setPNAoverBM3(id, rand());
      pnPrimDat.setFlag(id, rand());
      pnPrimDat.setVersions(id, 1, 1);

      i++;
    }
    cout << "Setting up ok" << endl;
    laserbluepulsedat.writeDB();
    cout << "1" << endl;
    laserirpulsedat.writeDB();
    cout << "2" << flush << endl;
    pnPrimDat.debug();
    pnPrimDat.setMaxDataToDump(2);
    pnPrimDat.writeDB();
    cout << "3" << endl;
  }

private:
  CondDBApp() = delete;  // hidden default constructor
  EcalCondDBInterface *econn;
  run_t run;
};

int main(int argc, char *argv[]) {
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
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
