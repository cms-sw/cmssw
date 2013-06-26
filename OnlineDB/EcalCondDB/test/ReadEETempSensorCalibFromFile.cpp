#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

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
    calitag.setGeneralTag("EE_P1_P2_P3");
    calitag.setMethod("LABORATORY");
    calitag.setVersion("FLOWER");
    calitag.setDataType("TEMP_SENSOR");


    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm since ;
    since.setToGMTime(1);   

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

  void testWrite(std::string filename)
  {

    cout << "read calibration file " << filename << endl;
    ifstream fin;
    fin.open(filename.c_str());

    std::string ch_line;
    // fin>> ch_line;


    cout << "Writing CaliCrystalIntercalDat objects to database..." << endl;
    cout << "Making a CaliIOV..." << flush;
    CaliIOV caliiov = this->makeCaliIOV();
    cout << "Done." << endl;

    this->printIOV(&caliiov);

    Tm eventTm = caliiov.getSince();
    CaliTag calitag = caliiov.getCaliTag();
    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSetOrdered("EE_readout_tower_zseccu", -1,1,1,9, 1, 50,"EE_readout_tower_zseccu",123);
    map<EcalLogicID, CaliTempDat> dataset;

    int z,sec,ccu,stato;
    float p1,p2,p3;
    int ir=-1;

    
    int fatto[600];
    for (int k=0; k<ecid_vec.size(); k++) {
      
      fatto[k]=-1;
      
    }
    

    while( fin.peek() != EOF )
      {
	fin >> z>>sec>>ccu>>p1>>p2>>p3>>stato;
	ir++;  
	
	cout << "z/sec/ccu/p1/p2/p3/stato="<<z<<"/"<<sec<<"/"<<ccu<<"/"<<p1<<"/"<<p2<<"/"<<p3<< endl;
	
	CaliTempDat capTemp;
	capTemp.setBeta(p1);
	capTemp.setR25(p2);
	capTemp.setOffset(p3);
	capTemp.setTaskStatus(stato);

       // Fill the dataset

	int kr=-1;
	for (int k=0; k<ecid_vec.size(); k++) {
	  
	  if(ecid_vec[k].getID1()==z && ecid_vec[k].getID2()==sec && ecid_vec[k].getID3()==ccu) kr=k;

	}
	if(kr!=-1){
	 dataset[ecid_vec[ kr ]] = capTemp;
	 fatto[kr]=1;
	}else {
	 cout<< "not filled DB for "<<kr<<endl; 
	}
	
      }

    fin.close();

    for (int k=0; k<ecid_vec.size(); k++) {
      
      if(fatto[k]==-1) {

	p1=  7.14E-10 ;
	p2=  -3.64E-04 ;
	p3=  5.39E+01 ;
	stato=0;
	CaliTempDat capTemp;
	capTemp.setBeta(p1);
	capTemp.setR25(p2);
	capTemp.setOffset(p3);
	capTemp.setTaskStatus(stato);
	dataset[ecid_vec[ k ]] = capTemp;
	
      }
      
    }



   // Insert the dataset, identifying by iov
   cout << "Inserting dataset..." << flush;
   econn->insertDataArraySet(&dataset, &caliiov);
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
  string filename;

  if (argc != 6) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <filename>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  filename = argv[5];

  try {
    CondDBApp app(sid, user, pass);

    app.testWrite(filename);
    //   app.testAllTables();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
