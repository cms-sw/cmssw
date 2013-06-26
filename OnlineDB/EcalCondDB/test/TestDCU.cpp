#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_dcu_types.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp( string sid, string user, string pass)
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



  DCUIOV makeDCUIOV()
  {
    LocationDef locdef;
    locdef.setLocation(locations[3]);
    DCUTag dcutag;
    dcutag.setLocationDef(locdef);


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
    DCUIOV dcuiov;

    dcuiov.setSince(since);
    dcuiov.setDCUTag(dcutag);
    
    return dcuiov;
  }



  /**
   *  Write objects with associated DCUIOVs
   *  IOVs are written using automatic updating of the 'till', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite()
  {
    cout << "Writing DCUCapsuleTempDat objects to database..." << endl;
    cout << "Making a DCUIOV..." << flush;
    DCUIOV dcuiov = this->makeDCUIOV();
    cout << "Done." << endl;

    this->printIOV(&dcuiov);

    Tm eventTm = dcuiov.getSince();
    DCUTag dcutag = dcuiov.getDCUTag();

    // Get channel ID for SM 10, crystal c      
    // int c = 1;
    //   EcalLogicID ecid;
    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", 10, 10, 1, 1700);
    //    ecid = econn->getEcalLogicID("EB_crystal_number", 10, c);



    map<EcalLogicID, DCUCapsuleTempDat> dataset;
    int count=0;
    for (int c=1; c<1701; c++){
      // the channels are turned in phi and eta 
      DCUCapsuleTempDat capTemp;
      
      int i = 1;
      float val = 0.11111 + i;
      capTemp.setCapsuleTemp(val);
      
      // Fill the dataset
      dataset[ecid_vec[count]] = capTemp;
      
      count++;
    }
    
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataArraySet(&dataset, &dcuiov);
    cout << "Done." << endl;

    // Fetch it back
    cout << "Fetching dataset..." << flush;
    dataset.clear();
    econn->fetchDataSet(&dataset, &dcuiov);
    cout << "retrieved " << dataset.size() << " channel-value pairs" << endl;
    printDataSet(&dataset);

    // Fetch back DCUIOV just written
    cout << "Fetching IOV just written..." << flush;
    DCUIOV dcuiov_prime = econn->fetchDCUIOV(&dcutag, eventTm);
    cout << "Done." << endl << endl << endl;
    this->printIOV(&dcuiov_prime);
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

    // DCUIOV tables
    DCUIOV dcuiov = this->makeDCUIOV();

    DCUCapsuleTempDat table01;
    testTable(&table01, &dcuiov, &logicID);

    DCUCapsuleTempRawDat table02;
    testTable(&table02, &dcuiov, &logicID);

    DCUIDarkDat table03;
    testTable(&table03, &dcuiov, &logicID);

    DCUIDarkPedDat table04;
    testTable(&table04, &dcuiov, &logicID);

    DCUVFETempDat table05;
    testTable(&table05, &dcuiov, &logicID);

    DCULVRTempsDat table06;
    testTable(&table06, &dcuiov, &logicID);

    DCULVRBTempsDat table07;
    testTable(&table07, &dcuiov, &logicID);

    DCULVRVoltagesDat table08;
    testTable(&table08, &dcuiov, &logicID);

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
  void printDataSet( const map<EcalLogicID, DCUCapsuleTempDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    DCUCapsuleTempDat capsTemp;

    int count = 0;
    typedef map< EcalLogicID, DCUCapsuleTempDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      capsTemp  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "capsule temp:           " << capsTemp.getCapsuleTemp() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }


  /**
   *   Print out a DCUTag
   */
  void printTag( const DCUTag* tag) const
  {
    cout << endl;
    cout << "=============DCUTag:" << endl;
    cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
    cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
    cout << "====================" << endl;
  }

  void printIOV( const DCUIOV* iov) const
  {
    cout << endl;
    cout << "=============DCUIOV:" << endl;
    DCUTag tag = iov->getDCUTag();
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
