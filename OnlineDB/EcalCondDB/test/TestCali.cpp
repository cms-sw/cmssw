#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_cali_types.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string sid, string user, string pass)
  {
    try {
      cout << "Making connection..." << flush;
      econn = new EcalCondDBInterface( sid, user, pass );
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



  CaliIOV makeCaliIOV()
  {
    LocationDef locdef;
    locdef.setLocation("LAB");
    CaliTag calitag;
    calitag.setLocationDef(locdef);


    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm since;
    since.setToCurrentGMTime();

    // Our beginning run number will be the seconds representation
    // of that time, and we will increment our IoV based on 
    // a microsecond representation
    uint64_t microseconds = since.microsTime();
    startmicros = microseconds;

    // Set the properties of the iov
    CaliIOV caliiov;

    caliiov.setSince(since);
    caliiov.setCaliTag(calitag);
    
    return caliiov;
  }



  /**
   *  Write objects with associated CaliIOVs
   *  IOVs are written using automatic updating of the 'till', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite()
  {
    cout << "Writing CaliCrystalIntercalDat objects to database..." << endl;
    cout << "Making a CaliIOV..." << flush;
    CaliIOV caliiov = this->makeCaliIOV();
    cout << "Done." << endl;

    this->printIOV(&caliiov);

    Tm eventTm = caliiov.getSince();
    CaliTag calitag = caliiov.getCaliTag();

    // Get channel ID for SM 10, crystal c      
    int c = 1;
    EcalLogicID ecid;
    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);

    // Set the data
    CaliCrystalIntercalDat crystalCal;;
    map<EcalLogicID, CaliCrystalIntercalDat> dataset;

    int i = 1;
    float val = 0.11111 + i;
    crystalCal.setCali(val);
    crystalCal.setCaliRMS(val);
    crystalCal.setNumEvents(i);
    crystalCal.setTaskStatus(true);

    // Fill the dataset
    dataset[ecid] = crystalCal;
    
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataSet(&dataset, &caliiov);
    cout << "Done." << endl;

    // Fetch it back
    cout << "Fetching dataset..." << flush;
    dataset.clear();
    econn->fetchDataSet(&dataset, &caliiov);
    cout << "retrieved " << dataset.size() << " channel-value pairs" << endl;
    printDataSet(&dataset);

    // Fetch back CaliIOV just written
    cout << "Fetching IOV just written..." << flush;
    CaliIOV caliiov_prime = econn->fetchCaliIOV(&calitag, eventTm);
    cout << "Done." << endl << endl << endl;
    this->printIOV(&caliiov_prime);
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

    // CaliIOV tables
    CaliIOV caliiov = this->makeCaliIOV();

    CaliGeneralDat table01;
    testTable(&table01, &caliiov, &logicID);

    CaliCrystalIntercalDat table02;
    testTable(&table02, &caliiov, &logicID);

    CaliHVScanRatioDat table03;
    testTable(&table03, &caliiov, &logicID);

    cout << "Test of writing to all tables complete" << endl;
    cout << tablesOK << " of " << tablesTried << " written to successfully" << endl << endl << endl;
  };



private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  int tablesTried;
  int tablesOK;

  /**
   *   Iterate through the dataset and print some data
   */
  void printDataSet( const map<EcalLogicID, CaliCrystalIntercalDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    CaliCrystalIntercalDat crystalCal;;

    int count = 0;
    typedef map< EcalLogicID, CaliCrystalIntercalDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      crystalCal  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "calibration:            " << crystalCal.getCali() << endl;
      cout << "calibration rms:        " << crystalCal.getCaliRMS() << endl;
      cout << "num events:             " << crystalCal.getNumEvents() << endl;
      cout << "task status:            " << crystalCal.getTaskStatus() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }


  /**
   *   Print out a CaliTag
   */
  void printTag( const CaliTag* tag) const
  {
    cout << endl;
    cout << "=============CaliTag:" << endl;
    cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
    cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
    cout << "Method:             " << tag->getMethod() << endl;
    cout << "Version:            " << tag->getVersion() << endl;
    cout << "Data Type:          " << tag->getDataType() << endl;
    cout << "====================" << endl;
  }

  void printIOV( const CaliIOV* iov) const
  {
    cout << endl;
    cout << "=============CaliIOV:" << endl;
    CaliTag tag = iov->getCaliTag();
    printTag(&tag);
    cout << "since:              " << iov->getSince().str() << endl;
    cout << "till:               " << iov->getTill().str() << endl;
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
    CondDBApp app(sid, user, pass);

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
