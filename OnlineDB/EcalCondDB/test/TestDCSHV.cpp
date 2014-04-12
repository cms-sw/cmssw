#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_od_types.h"


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


  void printHVDataSet( const map<EcalLogicID, RunDCSHVDat>* dataset, 
		     int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    RunDCSHVDat hv;

    int count = 0;
    typedef map< EcalLogicID, RunDCSHVDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      hv  = p->second;

      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "HV:                     " << hv.getHV() << endl;
      cout << "HV nominal:             " << hv.getHVNominal() << endl;
      cout << "HV status:              " << hv.getStatus() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }

  void printLVDataSet( const map<EcalLogicID, RunDCSLVDat>* dataset, 
		     int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    RunDCSLVDat lv;

    int count = 0;
    typedef map< EcalLogicID, RunDCSLVDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      lv  = p->second;

      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "LV:                     " << lv.getLV() << endl;
      cout << "LV nominal:             " << lv.getLVNominal() << endl;
      cout << "LV status:              " << lv.getStatus() << endl;
      cout << "========================" << endl;
    }
    cout << endl;
  }

  /**
   *  Write MonPedestalsDat objects with associated RunTag and RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testRead()
  {


    cout << "Reading Table " << endl;

    map<EcalLogicID, RunDCSHVDat> dataset;
    RunIOV *r = NULL;
    econn->fetchDataSet(&dataset, r);
    
    if (!dataset.size()) {
      throw(runtime_error("Zero rows read back"));
    }
    
    
    cout << "read OK" << endl;
    printHVDataSet(&dataset);

    map<EcalLogicID, RunDCSLVDat> dataset_lv;
    econn->fetchDataSet(&dataset_lv, r);
    
    if (!dataset_lv.size()) {
      throw(runtime_error("Zero rows read back"));
    }
    
    
    cout << "read OK" << endl;
    printLVDataSet(&dataset_lv);


    
    cout << "Done." << endl << endl << endl;
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
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <run_num> <config_tag>" 
	 << endl;
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

    app.testRead();

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
