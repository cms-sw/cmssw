#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TH2F.h"
#include "TF1.h"

using namespace std;

class CondDBApp {
public:
  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string host, string sid, string user, string pass, int port) {
    try {
      cout << "Making connection...to " << port << flush;
      econn = new EcalCondDBInterface(host, sid, user, pass, port);
      cout << "Done." << endl;
    } catch (runtime_error& e) {
      cerr << e.what() << endl;
      exit(-1);
    }
  }
  CondDBApp(string sid, string user, string pass) {
    try {
      cout << "Making connection...to " << sid << endl;
      econn = new EcalCondDBInterface(sid, user, pass);
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

  inline std::string to_string(char value[]) {
    std::ostringstream streamOut;
    streamOut << value;
    return streamOut.str();
  }

  void testReadOddWeights(int iconf_req) {
    // now we do something else
    // this is an example for reading the odd weights
    // for a given config iconf_req

    cout << "*****************************************" << endl;
    cout << "test reading odd weights with id=" << iconf_req << endl;
    cout << "*****************************************" << endl;

    FEConfigOddWeightInfo fe_wei_info;
    fe_wei_info.setId(iconf_req);
    econn->fetchConfigSet(&fe_wei_info);

    map<EcalLogicID, FEConfigOddWeightDat> dataset_wei;
    econn->fetchDataSet(&dataset_wei, &fe_wei_info);

    typedef map<EcalLogicID, FEConfigOddWeightDat>::const_iterator CIfewei;
    EcalLogicID ecid_xt;
    FEConfigOddWeightDat rd_wei;

    int rd_weiv[15176];
    for (int i = 0; i < 15176; i++) {
      rd_weiv[i] = 0;
    }
    int i = 0;
    for (CIfewei p = dataset_wei.begin(); p != dataset_wei.end(); p++) {
      ecid_xt = p->first;
      rd_wei = p->second;
      int sm_num = ecid_xt.getID1();
      int tow_num = ecid_xt.getID2();
      int strip_num = ecid_xt.getID3();
      rd_weiv[i] = rd_wei.getWeightGroupId();
      if (i < 10)
        std::cout << "here is the value for SM:" << sm_num << " tower:" << tow_num << " strip:" << strip_num
                  << " group id:" << rd_weiv[i] << endl;
      i = i + 1;
    }

    map<EcalLogicID, FEConfigOddWeightModeDat> dataset_mode;
    econn->fetchDataSet(&dataset_mode, &fe_wei_info);

    typedef map<EcalLogicID, FEConfigOddWeightModeDat>::const_iterator CIfem;
    FEConfigOddWeightModeDat rd_mode;

    int rd_modev[19] = {0};
    int k = 0;
    for (CIfem p = dataset_mode.begin(); p != dataset_mode.end(); p++) {
      rd_mode = p->second;
      rd_modev[0] = rd_mode.getEnableEBOddFilter();
      rd_modev[1] = rd_mode.getEnableEEOddFilter();
      // ...
      std::cout << "here is the value for the weight mode: EnableEBOddFilter:" << rd_modev[0]
                << " EnableEEOddFilter:" << rd_modev[1] << std::endl;
      k = k + 1;
    }

    cout << "*****************************************" << endl;
    cout << "test read done" << iconf_req << endl;
    cout << "*****************************************" << endl;
  }

  void testWriteOddWeights() {
    // now we do something else
    // this is an example for writing the odd weights

    cout << "*****************************************" << endl;
    cout << "************Inserting Odd weights************" << endl;
    cout << "*****************************************" << endl;

    FEConfigOddWeightInfo fe_wei_info;
    fe_wei_info.setNumberOfGroups(2);  // this eventually refers to some other table
    fe_wei_info.setConfigTag("my preferred odd weights");
    econn->insertConfigSet(&fe_wei_info);

    Tm tdb = fe_wei_info.getDBTime();
    //      tdb.dumpTm();

    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSet("EB_VFE", 1, 36, 1, 68, 1, 5);

    map<EcalLogicID, FEConfigOddWeightGroupDat> dataset;
    // we create 2 groups
    for (int ich = 0; ich < 2; ich++) {
      FEConfigOddWeightGroupDat wei;
      wei.setWeightGroupId(ich);
      if (ich == 0) {  // first group
        wei.setWeight0(0);
        wei.setWeight1(1);
        wei.setWeight2(2);
        wei.setWeight3(3);
        wei.setWeight4(4);
        wei.setWeight5(5);
      } else {  // second group
        wei.setWeight0(2);
        wei.setWeight1(3);
        wei.setWeight2(4);
        wei.setWeight3(5);
        wei.setWeight4(6);
        wei.setWeight5(7);
      }
      // Fill the dataset
      dataset[ecid_vec[ich]] = wei;  // we use any logic id, because it is in any case ignored...
    }

    // Insert the dataset
    econn->insertDataArraySet(&dataset, &fe_wei_info);

    vector<EcalLogicID> my_StripEcalLogicId1_EE;
    vector<EcalLogicID> my_StripEcalLogicId2_EE;

    // EE Strip identifiers
    // DCC=601-609 TT = ~40 EEstrip = 5
    my_StripEcalLogicId1_EE =
        econn->getEcalLogicIDSetOrdered("ECAL_readout_strip", 601, 609, 1, 100, 0, 5, "ECAL_readout_strip", 123);
    // EE Strip identifiers
    // DCC=646-654 TT = ~40 EEstrip = 5
    my_StripEcalLogicId2_EE =
        econn->getEcalLogicIDSetOrdered("ECAL_readout_strip", 646, 654, 1, 100, 0, 5, "ECAL_readout_strip", 123);

    // now we store in the DB the correspondence btw channels and odd weight groups
    map<EcalLogicID, FEConfigOddWeightDat> dataset2;
    // in this case I decide in a stupid way which channel belongs to which group
    for (int ich = 0; ich < (int)ecid_vec.size(); ich++) {
      FEConfigOddWeightDat weid;
      if (ecid_vec[ich].getID1() <= 609 || ecid_vec[ich].getID1() > 645) {
        weid.setWeightGroupId(0);  // EB
      }
      // Fill the dataset
      dataset2[ecid_vec[ich]] = weid;
    }

    // EE loop
    for (int ich = 0; ich < (int)my_StripEcalLogicId1_EE.size(); ich++) {
      FEConfigOddWeightDat wut;
      int igroup = 1;  // this group is for EE
      wut.setWeightGroupId(igroup);
      // Fill the dataset
      dataset2[my_StripEcalLogicId1_EE[ich]] = wut;
    }
    // EE loop 2 (we had to split the ids of EE in 2 vectors to avoid crash!)
    for (int ich = 0; ich < (int)my_StripEcalLogicId2_EE.size(); ich++) {
      FEConfigOddWeightDat wut;
      int igroup = 1;  // this group is for EE
      wut.setWeightGroupId(igroup);
      // Fill the dataset
      dataset2[my_StripEcalLogicId2_EE[ich]] = wut;
    }

    // Insert the dataset

    econn->insertDataArraySet(&dataset2, &fe_wei_info);

    map<EcalLogicID, FEConfigOddWeightModeDat> datasetmode;
    FEConfigOddWeightModeDat wei_mode;
    wei_mode.setEnableEBOddFilter(3);
    wei_mode.setDisableEBEvenPeakFinder(4);
    wei_mode.setFenixEEStripOutput(5);
    // this is just a test, I leave the other parameters at 0

    // Fill the dataset
    datasetmode[ecid_vec[0]] = wei_mode;  // we use any logic id, because it is in any case ignored...

    econn->insertDataSet(&datasetmode, &fe_wei_info);

    cout << "*****************************************" << endl;
    cout << "*********** odd weights done ************" << endl;
    cout << "*****************************************" << endl;
  }

private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;

  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  TFile* f;
  TH2F* mataq_vs_run;
  TH2F* apd_pn_mean_vs_run;

  void printTag(const RunTag* tag) const {
    cout << endl;
    cout << "=============RunTag:" << endl;
    cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
    cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
    cout << "Run Type:           " << tag->getRunTypeDef().getRunType() << endl;
    cout << "====================" << endl;
  }

  void printIOV(const RunIOV* iov) const {
    cout << endl;
    cout << "=============RunIOV:" << endl;
    RunTag tag = iov->getRunTag();
    printTag(&tag);
    cout << "Run Number:         " << iov->getRunNumber() << endl;
    cout << "Run Start:          " << iov->getRunStart().str() << endl;
    cout << "Run End:            " << iov->getRunEnd().str() << endl;
    cout << "====================" << endl;
  }
};

int main(int argc, char* argv[]) {
  string sid;
  string user;
  string pass;
  string cfg_str;
  string read_and_w_str;

  if (argc != 6) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <cfg_id> <read_or_write(0:read 1:write 2:read_and_write)>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  cfg_str = argv[4];
  int cfg_id = atoi(cfg_str.c_str());
  read_and_w_str = argv[5];
  int rw_id = atoi(read_and_w_str.c_str());

  try {
    CondDBApp app(sid, user, pass);
    if (rw_id == 1 || rw_id == 2)
      app.testWriteOddWeights();
    if (rw_id == 0 || rw_id == 2)
      app.testReadOddWeights(cfg_id);
  } catch (exception& e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
