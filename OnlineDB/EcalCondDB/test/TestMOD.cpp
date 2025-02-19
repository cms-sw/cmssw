#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_mod_types.h"


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

  
  MODRunIOV makeMODRunIOV(RunIOV* runiov)
  {
    MODRunIOV modiov;
    modiov.setRunIOV(*runiov);
    modiov.setSubRunNumber(0);
    modiov.setSubRunStart(runiov->getRunStart());

    return modiov;
  }



  /**
   *  Write MODCCSTRDat objects with associated RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite()
  {
    cout << "Writing MODCCSTRDat to database..." << endl;
    RunIOV runiov = this->makeRunIOV();
    RunTag runtag = runiov.getRunTag();
    run_t run = runiov.getRunNumber();

    // write to the DB
    cout << "Inserting run..." << flush;
    econn->insertRunIOV(&runiov);
    cout << "Done." << endl;
    printIOV(&runiov);

    // fetch it back
    cout << "Fetching run by tag just used..." << flush;
    RunIOV runiov_prime = econn->fetchRunIOV(&runtag, run);
    cout << "Done." << endl;
    printIOV(&runiov_prime);
    cout << "Fetching run by location..." << flush;
    RunIOV runiov_prime2 = econn->fetchRunIOV(runtag.getLocationDef().getLocation(), run);
    cout << "Done." << endl;
    printIOV(&runiov_prime2);    

    // MODitoring Tag and IOV
    MODRunIOV modiov = this->makeMODRunIOV(&runiov);

    // Get channel ID for SM 10, crystal c      
    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("EB_token_ring", 10, c);

    // Set the data
    /*    MODCCSTRDat d;
    map<EcalLogicID, MODCCSTRDat> dataset;
    int i = 1112287340;
    d.setWord(i);
    dataset[ecid] = d;
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset, &modiov);
    cout << "Done." << endl;
    */

    // Set the data
    MODCCSHFDat dhf;
    map<EcalLogicID, MODCCSHFDat> datasethf;
    std::string fname="/u1/fra/test.txt";
    dhf.setFile(fname);
    dhf.setTest(123);
    std::cout<< "here it is: " <<dhf.getFile()<< endl;
    datasethf[ecid] = dhf;
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&datasethf, &modiov);
    cout << "Done." << endl;

    // Fetch it back
    /*    cout << "Fetching dataset..." << flush;
    dataset.clear();
    econn->fetchDataSet(&dataset, &modiov);
    cout << "retrieved " << dataset.size() << " channel-value pairs" << endl;
    */

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


  void printIOV( const RunIOV* iov) const
  {
    cout << endl;
    cout << "=============RunIOV:" << endl;
    RunTag tag = iov->getRunTag();

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
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
