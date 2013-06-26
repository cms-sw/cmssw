#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_dcu_types.h"


using namespace std;
using std::string;

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
    locdef.setLocation("P5_Co");
    DCUTag dcutag;
    dcutag.setLocationDef(locdef);
    dcutag.setGeneralTag("IDarkPedestalsRun");

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

  void testWrite(std::string filename)
  {

    float temp_vec[61200];
    int ih4_vec[61200];
    int ih4;
    float temp;

    for(int ic=0; ic<61200; ic++){
      temp_vec[ic]=0;
      ih4_vec[ic]=0;
    }


    cout << "read IDark Pedestal file " << filename << endl;


    FILE *fin; // input file
    fin = fopen(filename.c_str(),"r");

    char line[256];

    fgets(line,255,fin); // first is the comment line 


    while(fgets(line,255,fin)) 
      {

	std::string EBorEE;
	std::stringstream aStrStream;
	aStrStream << line;

	int ism=0;
	aStrStream >> EBorEE >> ism>> ih4 >> temp;

	ih4=ih4-1; // in the file ih4 follows the electronics numbering but plus one

	if (EBorEE == "EB-") ism=ism+18; 

	int ic=(ism-1)*1700+ih4;

        temp_vec[ic]=temp;
        ih4_vec[ic]=ih4;

      }
    fclose(fin);



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

    cout << "Loading ecal logic id from DB" << endl; 
    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_elec_crystal_number", 1, 36, 0, 1699,
					       EcalLogicID::NULLID,EcalLogicID::NULLID,
					       "EB_elec_crystal_number",12);


    map<EcalLogicID, DCUIDarkPedDat> dataset;

    
    for (int c=0; c<61200; c++){
      // the channels are turned in phi and eta 
      
      DCUIDarkPedDat capTemp;
      capTemp.setPed(temp_vec[c]);
      // Fill the dataset
      dataset[ecid_vec[c]] = capTemp;

    }
    
    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataArraySet(&dataset, &dcuiov);
    cout << "Done." << endl;

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
  string filename;
  string sid;
  string user;
  string pass;


  if (argc != 5) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <file>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  filename=argv[4];

  try {
    CondDBApp app(sid, user, pass);

    app.testWrite(filename);

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
