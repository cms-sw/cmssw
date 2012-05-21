#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <cstdlib>
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
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
  }

  /**
   *  App destructor;  Cleans up database connection
   */
  ~CondDBApp() 
  {
    delete econn;
  }

  int testWrite()
  {
    int las_id = -1;
    cout << "Writing a laser configuration to database..." << endl;

    //Laser
    ODLaserConfig las ;
    las.setDebug();
    las.setConfigTag("LasConfig_1");
    las.setMatacqBaseAddress(76523);
    las.setMatacqMode("whatever");
    las.setMatacqNone(-1);
    las.setChannelMask(3);
    las.setMaxSamplesForDaq("10");
    las.setMatacqFedId(67);
    las.setPedestalFile("thisfile");
    las.setUseBuffer(1);
    las.setPostTrig(2);
    las.setFPMode(3);
    las.setHalModuleFile("bho");
    las.setMatacqVernierMax(100);
    las.setMatacqVernierMin(70);
    las.setHalAddressTableFile("bho");
    las.setHalStaticTableFile("bho");
    las.setMatacqSerialNumber("TX-738420");
    las.setPedestalRunEventCount(24);
    las.setRawDataMode(56);
    las.setMatacqAcquisitionMode("testing");
    las.setLocalOutputFile("giovanni.organtini");
    las.setEMTCNone(0);
    las.setWTE2LaserDelay(3);
    las.setLaserPhase(15);
    las.setEMTCTTCIn(23);
    las.setEMTCSlotId(67);
    las.setWaveLength(425);
    las.setPower(5);
    las.setOpticalSwitch( 6 );
    las.setFilter( 7 );
    las.setLaserControlOn(43);
    las.setLaserControlHost("localhost");
    las.setLaserControlPort(80);
    las.setWTE2LedDelay(125);
    las.setLed1ON(1);
    las.setLed2ON(2);
    las.setLed3ON(3);
    las.setLed4ON(4);
    las.setVinj(422);
    las.setOrangeLedMonAmpl(565);
    las.setBlueLedMonAmpl(522);
    las.setTrigLogFile("trig.log");
    las.setLedControlON(4);
    las.setLedControlHost("celeste.aida");
    las.setLedControlPort(1965);
    las.setIRLaserPower(5263);
    las.setGreenLaserPower(63723);
    las.setRedLaserPower(29384);
    las.setBlueLaserLogAttenuator(9);
    las.setIRLaserLogAttenuator(8);
    las.setGreenLaserLogAttenuator(7);
    las.setRedLaserLogAttenuator(6);
    las.setLaserConfigFile("gioachino.rossini");
    unsigned char clob[27] = "abcdefghijklmnopqrstuvwxyz";
    las.setLaserClob(&clob[0], 26);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&las);
    las_id=las.getId();
    cout << "LASER Config inserted with ID "<< las_id<< endl;
    return las_id;
  };

  int testRead(int id) {
    ODLaserConfig las;
    std::cout << "Reading back Laser Configuration ID: " << id << std::endl;
    las.setId(id);
    econn->fetchConfigSet(&las);
    std::cout << "Read back Laser Configuration ID: " << id 
	      << "...printing it out" << std::endl << std::flush;
    las.printout();
    return id;
  };

private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  string locations[4];
  uint64_t startmicros;
  uint64_t endmicros;
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

  if ((int)sid.find("INT2R") < 0) {
    cout << "Use this application only on INT2R" << endl;
    exit(-1);
  }

  try {
    CondDBApp app(host, sid, user, pass);

    int id = app.testWrite();
    app.testRead(id);

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
