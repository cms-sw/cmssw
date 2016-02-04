#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemChErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemTTErrorsDat.h"
#include "CondTools/Ecal/interface/EcalErrorDictionary.h"


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
      econn = new EcalCondDBInterface(  sid, user, pass );
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



  void testWrite()
  {
    cout << "Writing RunCrystalErrorsDat object to database..." << endl;
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

    // Get channel ID for SM 10, crystal c      
    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);

    // Set the data
    RunCrystalErrorsDat cryerr;
    map<EcalLogicID, RunCrystalErrorsDat> dataset;

    uint64_t bits = 0;
    bits = 0;
    bits |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
    bits |= EcalErrorDictionary::getMask("CH_ID_ERROR");
    bits |= EcalErrorDictionary::getMask("LASER_MEAN_WARNING");
    cryerr.setErrorBits(bits);

    // Fill the dataset
    dataset[ecid] = cryerr;
    
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset, &runiov);
    cout << "Done." << endl;

    // Fetch it back
    cout << "Fetching dataset..." << flush;
    dataset.clear();
    econn->fetchDataSet(&dataset, &runiov);
    cout << "retrieved " << dataset.size() << " channel-value pairs" << endl;
    printDataSet(&dataset);
  };



  void testLookup()
  {
    cout << "Testing Lookup of RunCrystalErrorsDat" << endl;
    RunIOV runiov = this->makeRunIOV();
    RunTag runtag = runiov.getRunTag();
    runtag.setGeneralTag("testLookup " + runiov.getRunStart().str() );  // XXX hack for testing
    runiov.setRunTag(runtag);

    run_t run = runiov.getRunNumber();
    uint64_t micros = runiov.getRunStart().microsTime();

    // Get channel ID for SM 10, crystal c      
    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);
    
    // Prepare data objects
    RunCrystalErrorsDat cryerr;
    map<EcalLogicID, RunCrystalErrorsDat> dataset;
    srand(time(0));

    // Write 12 runs to the DB, and write data every 4
    for (int i = 0;  i < 12; i++) {
      cout << "i:  " << i << "  Inserting run " << run << "..." << flush;
      econn->insertRunIOV(&runiov);

      if (i%4 == 0) {
	// Fill the dataset
	cryerr.setErrorBits((uint64_t)rand() * (uint64_t)rand());
	dataset[ecid] = cryerr;
	
	// Insert the dataset, identifying by iov
	cout << "Inserting dataset..." << flush;
	econn->insertDataSet(&dataset, &runiov);
      }
      cout << "Done." << endl;

      // Increment RunIOV
      run++;
      micros += 100000 * 60 * 60;  // 1 hour
      runiov.setRunNumber(run);
      runiov.setRunStart(Tm(micros));
    }

    // Now pretend we don't know what runs data was inserted for
    // Lookup data based on some runs (every 3 in this case)
    RunIOV validIOV;
    run_t somerun = startrun;

    for (int i = 0; i < 4; i++) {
      cout << "Fetching dataset for run: " << somerun << "..." << endl;
      econn->fetchValidDataSet(&dataset, &validIOV, &runtag, somerun);
      cout << "Attached to run: " << validIOV.getRunNumber() << endl;
      printDataSet(&dataset);
      somerun += 3;
    }
  }

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
    RunCrystalErrorsDat table01;
    testTable(&table01, &runiov, &logicID);

    RunTTErrorsDat table02;
    testTable(&table02, &runiov, &logicID);

    RunPNErrorsDat table03;
    testTable(&table03, &runiov, &logicID);

    RunMemChErrorsDat table04;
    testTable(&table04, &runiov, &logicID);

    RunMemTTErrorsDat table05;
    testTable(&table05, &runiov, &logicID);

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
  void printDataSet( const map<EcalLogicID, RunCrystalErrorsDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    RunCrystalErrorsDat cryerr;

    int count = 0;
    typedef map< EcalLogicID, RunCrystalErrorsDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      cryerr  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "Error Bits:             " 
	   << "0x" << hex << setfill('0') << setw(16) << cryerr.getErrorBits() 
	   << " (" << dec << setfill(' ') << setw(20) << cryerr.getErrorBits() << ")" << endl;
      cout << "Errors:" << endl;
      EcalErrorDictionary::printErrors(cryerr.getErrorBits());
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
    app.testLookup();

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
