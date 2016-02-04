#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>

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


  RunIOV makeRunIOV(int run_num)
  {
    // The objects necessary to identify a dataset
    LocationDef locdef;
    RunTypeDef rundef;
    RunTag runtag;

    locdef.setLocation("P5_Co");

    rundef.setRunType("TEST");
    
    runtag.setLocationDef(locdef);
    runtag.setRunTypeDef(rundef);

    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm startTm;
    startTm.setToCurrentGMTime();

    uint64_t microseconds = startTm.microsTime();
    startmicros = microseconds;

    int startrun = run_num;

    cout << "Starting Time:    " << startTm.str() << endl;
    cout << "Starting Micros:  " << startmicros << endl;
    cout << "Starting Run:     " << startrun << endl;

    // Set the properties of the iov
    RunIOV runiov;

    startTm.setToMicrosTime(microseconds);
    cout << "Setting run " << startrun << " run_start " << startTm.str() << endl;
    runiov.setRunNumber(startrun);
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
  void testWrite(int run_num, std::string config_tag)
  {
    cout << "Writing MonPedestalsDatObjects to database..." << endl;
    RunIOV runiov = this->makeRunIOV(run_num);
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




    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("ECAL");

    // recuperare last version !!!

    FEConfigMainInfo cfg_info;
    cfg_info.setConfigTag(config_tag);
    econn->fetchConfigSet(&cfg_info);

    int iversion= cfg_info.getVersion();


    // Set the data
    RunTPGConfigDat cfgdat;
    map<EcalLogicID, RunTPGConfigDat> dataset;

    cfgdat.setConfigTag(config_tag);
    cfgdat.setVersion(iversion);

    // Fill the dataset
    dataset[ecid] = cfgdat;

    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset, &runiov);
    cout << "Done." << endl;



    sleep(3);






    cout << "updating start time..." << flush;

    Tm startTm;
    startTm.setToCurrentGMTime();
    uint64_t microseconds = startTm.microsTime();
    startTm.setToMicrosTime(microseconds);
    runiov.setRunStart(startTm);

    // write to the DB
    cout << "Inserting run..." << flush;
    econn->updateRunIOVStartTime(&runiov);
    cout << "Done." << endl;
    printIOV(&runiov);


    sleep(3);

    cout << "updating end time..." << flush;

    Tm endTm;
    endTm.setToCurrentGMTime();
    microseconds = endTm.microsTime();
    endTm.setToMicrosTime(microseconds);
    runiov.setRunEnd(endTm);

    // write to the DB
    cout << "Inserting run..." << flush;
    econn->updateRunIOVEndTime(&runiov);
    cout << "Done." << endl;
    printIOV(&runiov);

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
  int run_num;
  string config_tag;


  if (argc != 7) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  run_num = atoi(argv[5]);
  config_tag = argv[6];


  try {
    CondDBApp app(host, sid, user, pass);

    app.testWrite(run_num, config_tag);

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
