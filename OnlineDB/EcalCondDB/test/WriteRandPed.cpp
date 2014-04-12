#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"


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
      if (host != "?") {
	econn = new EcalCondDBInterface(host, sid, user, pass );
      } else {
	econn = new EcalCondDBInterface( sid, user, pass );
      }
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


  /**
   *  Write MonPedestalsDat objects with associated RunTag and RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite(int startrun, int numruns)
  {
    // initial run
    run_t run = startrun;

    cout << "Writing MonPedestalsDat Objects to database..." << endl;
    // The objects necessary to identify a dataset
    LocationDef locdef;
    RunTypeDef rundef;
    RunTag runtag;
    RunIOV runiov;

    MonVersionDef monverdef;
    MonRunTag montag;
    MonRunIOV moniov;

    Tm startTm;
    Tm endTm;
    uint64_t oneMin = 1 * 60 * 1000000;
    uint64_t twentyMin = 20 * 60 * 1000000;

    // Set the beginning time
    startTm.setToString("2007-01-01 00:00:00");
    uint64_t microseconds = startTm.microsTime();
    microseconds += (startrun-1)*twentyMin;

    // The number of channels and IOVs we will write

    // The objects necessary to define a dataset
    EcalLogicID ecid;
    MonPedestalsDat ped;
    map<EcalLogicID, MonPedestalsDat> dataset;

    // Set the properties of the tag
    // The objects necessary to identify a dataset
    locdef.setLocation("H4");
    rundef.setRunType("PEDESTAL");
    
    runtag.setLocationDef(locdef);
    runtag.setRunTypeDef(rundef);

    runiov.setRunTag(runtag);

    monverdef.setMonitoringVersion("test01");
    montag.setMonVersionDef(monverdef);

    moniov.setMonRunTag(montag);

    // Get a set of channels to use
    cout << "Getting channel set..." << flush;
    vector<EcalLogicID> channels;
    channels = econn->getEcalLogicIDSet("EB_crystal_number",
					1, 36,   // SM
					1, 1700 // crystal
					);
    cout << "Done." << endl;

    cout << "Writing " << numruns << " sets of pedestals..." << endl;
    for (int i=1; i<=numruns; i++) {  // For every IOV we will write
      // Set the properties of the runiov
      startTm.setToMicrosTime(microseconds);
      endTm.setToMicrosTime(microseconds + oneMin);
      runiov.setRunNumber(run);
      runiov.setRunStart(startTm);
      runiov.setRunEnd(endTm);

      // Write runiov
      econn->insertRunIOV(&runiov);

      // Set the properties of the moniov
      moniov.setRunIOV(runiov);
      moniov.setSubRunNumber(0);
      moniov.setSubRunStart(startTm);
      moniov.setSubRunEnd(endTm);

      for (vector<EcalLogicID>::const_iterator p = channels.begin();
	   p != channels.end();
	   ++p) {
      
	// Set the data
	float val = 1 + rand();
	ped.setPedMeanG1(val);
	ped.setPedMeanG6(val);
	ped.setPedMeanG12(val);
	ped.setPedRMSG1(val);
	ped.setPedRMSG6(val);
	ped.setPedRMSG12(val);
	ped.setTaskStatus(1);
	
	// Fill the dataset
	dataset[*p] = ped;
      }

      // Insert the dataset, identifying by moniov
      cout << "Writing IOV " << i << " of " << numruns 
	   << " (run " << run << ")..." << flush;
      econn->insertDataArraySet(&dataset, &moniov );
      cout << "Done." << endl;

      // Increment IOV run, start_time
      run++;  // one run
      microseconds += twentyMin;
    }
    cout << "Done." << endl << endl;
  }



private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
};



int main (int argc, char* argv[])
{
  string host;
  string sid;
  string user;
  string pass;
  int startrun;
  int numruns;

  if (argc != 7) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <start run> <num runs>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  startrun = atoi(argv[5]);
  numruns = atoi(argv[6]);

  try {
    cout << "Host:  " << host << endl;
    cout << "SID:   " << sid << endl;
    cout << "User:  " << user << endl;
    
    CondDBApp app(host, sid, user, pass);

    app.testWrite(startrun, numruns);
  } catch (runtime_error &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
