#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string host, string sid, string user, string pass)
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


  RunIOV makeRunIOV()
  {
    // The objects necessary to identify a dataset
    LocationDef locdef;
    RunTypeDef rundef;
    RunTag runtag;

    locdef.setLocation(locations[3]);

    rundef.setRunType("TEST");
    
    runtag.setLocationDef(locdef);
    runtag.setRunTypeDef(rundef);

    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm startTm;
    startTm.setToCurrentGMTime();

    // Our beginning run number will be the seconds representation
    // of that time, and we will increment our IoV based on 
    // a microsecond representation
    uint64_t microseconds = startTm.microsTime();
    startmicros = microseconds;
    run_t run = (int)(microseconds/1000000);
    startrun = run;

    cout << "Starting Time:    " << startTm.str() << endl;
    cout << "Starting Micros:  " << startmicros << endl;
    cout << "Starting Run:     " << startrun << endl;

    // Set the properties of the iov
    RunIOV runiov;

    startTm.setToMicrosTime(microseconds);
    cout << "Setting run " << run << " run_start " << startTm.str() << endl;
    runiov.setRunNumber(run);
    runiov.setRunStart(startTm);
    runiov.setRunTag(runtag);
    
    return runiov;
  }

  
  MonRunIOV makeMonRunIOV(RunIOV* runiov)
  {
    // Monitoring Tag and IOV
    MonVersionDef monverdef;
    monverdef.setMonitoringVersion("test01");

    MonRunTag montag;
    montag.setMonVersionDef(monverdef);

    MonRunIOV moniov;
    moniov.setMonRunTag(montag);
    moniov.setRunIOV(*runiov);
    moniov.setSubRunNumber(0);
    moniov.setSubRunStart(runiov->getRunStart());

    return moniov;
  }



  /**
   *  Write MonPedestalsDat objects with associated RunTag and RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite()
  {
    cout << "Writing MonPedestalsDatObjects to database..." << endl;
    RunIOV runiov = this->makeRunIOV();
    RunTag runtag = runiov.getRunTag();
    run_t run = runiov.getRunNumber();

    // write to the DB
    cout << "Inserting run..." << flush;
    econn->insertRunIOV(&runiov);
    cout << "Done." << endl;
    printIOV(&runiov);

//     // write another run with the same run number to generate exception later
//     runtag.setGeneralTag("different");
//     runiov.setRunTag(runtag);
//     econn->insertRunIOV(&runiov);


    // fetch it back
    cout << "Fetching run by tag just used..." << flush;
    RunIOV runiov_prime = econn->fetchRunIOV(&runtag, run);
    cout << "Done." << endl;
    printIOV(&runiov_prime);
    cout << "Fetching run by location..." << flush;
    RunIOV runiov_prime2 = econn->fetchRunIOV(runtag.getLocationDef().getLocation(), run);
    cout << "Done." << endl;
    printIOV(&runiov_prime2);    

    // Monitoring Tag and IOV
    MonRunIOV moniov = this->makeMonRunIOV(&runiov);

    // Get channel ID for SM 10, crystal c      
    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);

    // Set the data
    MonPedestalsDat ped;
    map<EcalLogicID, MonPedestalsDat> dataset;

    int i = 1;
    float val = 0.11111 + i;
    ped.setPedMeanG1(val);
    ped.setPedMeanG6(val);
    ped.setPedMeanG12(val);
    ped.setPedRMSG1(val);
    ped.setPedRMSG6(val);
    ped.setPedRMSG12(val);
    ped.setTaskStatus(1);
    
    // Fill the dataset
    dataset[ecid] = ped;
    
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset, &moniov);
    cout << "Done." << endl;

    // Fetch it back
    cout << "Fetching dataset..." << flush;
    dataset.clear();
    econn->fetchDataSet(&dataset, &moniov);
    cout << "retrieved " << dataset.size() << " channel-value pairs" << endl;
    printDataSet(&dataset);

    // Fetch back MonRunIOV just written
    MonRunTag montag = moniov.getMonRunTag();
    subrun_t subrun = moniov.getSubRunNumber();
    cout << "Fetching subrun..." << flush;
    MonRunIOV monruniov_prime = econn->fetchMonRunIOV(&runtag, &montag, run, subrun);
    cout << "Done." << endl << endl << endl;
  };



  /**
   *  Write a data set
   */
  template<class DATT, class IOVT>
  void testTable(DATT* data, IOVT* iov, EcalLogicID* ecid)
  {
    tablesTried++;
    try {
      cout << "Table " << tablesTried << "..." << flush;
      map<EcalLogicID, DATT> dataset;
      dataset[*ecid] = *data;
      econn->insertDataSet(&dataset, iov);
      cout << "write OK..." << flush;
      dataset.clear();
      econn->fetchDataSet(&dataset, iov);
      if (!dataset.size()) {
	throw(runtime_error("Zero rows read back"));
      }
      cout << "read OK" << endl;
      tablesOK++;
    } catch (runtime_error &e) {
      cout << "testTable FAILED:  " << e.what() << endl;
    } catch (...) {
      cout << "testTable FAILED:  unknown exception" << endl;
    }
  }



  /**
   *  Write to each of the data tables
   */
  void testAllTables()
  {
    cout << "Testing writing to all tables..." << endl;
    tablesTried = 0;
    tablesOK = 0;
    
    // get a dummy logic ID
    EcalLogicID logicID = econn->getEcalLogicID(-1);

    // RunIOV tables
    RunIOV runiov = this->makeRunIOV();

    // tables using RunIOV
    RunDat table01;
    testTable(&table01, &runiov, &logicID);

    RunConfigDat table01a;
    testTable(&table01a, &runiov, &logicID);

    RunH4TablePositionDat table01b;
    testTable(&table01b, &runiov, &logicID);

    // MonRunIOV tables
    MonRunIOV moniov = this->makeMonRunIOV(&runiov);

    MonPedestalsDat table02;
    testTable(&table02, &moniov, &logicID);

    MonRunOutcomeDef monRunOutcomeDef;
    monRunOutcomeDef.setShortDesc("success");
    MonRunDat table03;
    table03.setMonRunOutcomeDef(monRunOutcomeDef);
    testTable(&table03, &moniov, &logicID);

    MonTestPulseDat table04;
    testTable(&table04, &moniov, &logicID);

    MonPulseShapeDat table05;
    testTable(&table05, &moniov, &logicID);

    MonDelaysTTDat table06;
    testTable(&table06, &moniov, &logicID);

    MonShapeQualityDat table07;
    testTable(&table07, &moniov, &logicID);

    MonPedestalsOnlineDat table08;
    testTable(&table08, &moniov, &logicID);

    MonPedestalOffsetsDat table09;
    testTable(&table09, &moniov, &logicID);

    MonCrystalConsistencyDat table10;
    testTable(&table10, &moniov, &logicID);

    MonTTConsistencyDat table11;
    testTable(&table11, &moniov, &logicID);

    MonOccupancyDat table12;
    testTable(&table12, &moniov, &logicID);

    MonLaserBlueDat table14;
    testTable(&table14, &moniov, &logicID);

    MonLaserGreenDat table15;
    testTable(&table15, &moniov, &logicID);

    MonLaserRedDat table16;
    testTable(&table16, &moniov, &logicID);

    MonLaserIRedDat table17;
    testTable(&table17, &moniov, &logicID);

    MonPNBlueDat table18;
    testTable(&table18, &moniov, &logicID);

    MonPNGreenDat table19;
    testTable(&table19, &moniov, &logicID);
    
    MonPNRedDat table20;
    testTable(&table20, &moniov, &logicID);
    
    MonPNIRedDat table21;
    testTable(&table21, &moniov, &logicID);

    MonPNMGPADat table22;
    testTable(&table22, &moniov, &logicID);

    MonPNPedDat table23;
    testTable(&table23, &moniov, &logicID);

    MonMemChConsistencyDat table25;
    testTable(&table25, &moniov, &logicID);

    MonMemTTConsistencyDat table26;
    testTable(&table26, &moniov, &logicID);

    MonH4TablePositionDat table27;
    testTable(&table27, &moniov, &logicID);

    MonLaserPulseDat table29;
    testTable(&table29, &moniov, &logicID);

    cout << "Test of writing to all tables complete" << endl;
    cout << tablesOK << " of " << tablesTried << " written to successfully" << endl << endl << endl;
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
   *   Iterate through the dataset and print some data
   */
  void printDataSet( const map<EcalLogicID, MonPedestalsDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    MonPedestalsDat ped;

    int count = 0;
    typedef map< EcalLogicID, MonPedestalsDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      ped  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "Mean G1:                " << ped.getPedMeanG1() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }


  /**
   *   Print out a RunTag
   */
  void printTag( const RunTag* tag) const
  {
    cout << endl;
    cout << "=============RunTag:" << endl;
    cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
    cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
    cout << "Run Type:           " << tag->getRunTypeDef().getRunType() << endl;
    cout << "====================" << endl;
  }

  void printIOV( const RunIOV* iov) const
  {
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
    CondDBApp app(host, sid, user, pass);

    app.testWrite();
    app.testAllTables();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
