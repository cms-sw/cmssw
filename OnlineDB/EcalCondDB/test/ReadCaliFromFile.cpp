#include <iostream>

#include <fstream>
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
    calitag.setDataType("TEMP_SENSOR");
    calitag.setMethod("COSMICS");
    calitag.setVersion("OFFSET");
    calitag.setGeneralTag("EB_with_offset_in_situ");

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

    //-------------------------------------------------------------
    int convertFromConstructionSMToSlot(int sm_constr,int sm_slot){
      // input either cosntruction number or slot number and returns the other
      // the slots are numbered first EB+ slot 1 ...18 then EB- 1... 18
      // the slots start at 1 and the SM start at 0
      //-------------------------------------------------------------
      int slot_to_constr[37]={-1,12,17,10,1,8,4,27,20,23,25,6,34,35,15,18,30,21,9
			      ,24,22,13,31,26,16,2,11,5,0,29,28,14,33,32,3,7,19};
      int constr_to_slot[36]={28,4,25,34,6,27,11,35,5,18,3,26,1,21,31,14,24,2,15,
			      36,8,17,20,9,19,10,23,7,30,29,16,22,33,32,12,13 };

      int result=0;
      if(sm_constr!=-1) {
	result=constr_to_slot[sm_constr];
      } else if(sm_slot!=-1) {
	result=slot_to_constr[sm_slot];
      }
      return result;
    }


  /**
   *  Write objects with associated CaliIOVs
   *  IOVs are written using automatic updating of the 'till', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite(std::string filename)
  {

    cout << "read calibration file " << filename << endl;
    ifstream fin;
    fin.open(filename.c_str());

    std::string ch_line;
    // fin>> ch_line;

    float temp_vec[61200];
    float beta_vec[61200];
    float r25_vec[61200];
    float nic_vec[61200];
    for(int ic=0; ic<61200; ic++){
      beta_vec[ic]=0;
      r25_vec[ic]=0;
      temp_vec[ic]=0;
      nic_vec[ic]=0;
    }


    int sm,i;
    float beta, r25, nic;
    // int slot_num=0;
    while( fin.peek() != EOF )
      {

	//	fin >> sm>>i>>j>>ih4>>temp>>beta>>r25>>nic;
	fin >> sm>>i>>beta>>r25>>nic;
	  
	
	if(i<10) cout << "sm/i/beta/r25/offset="<<sm<<"/"<<i<<"/"<<beta<<"/"<<r25<<"/"<<nic<< endl;
	//	slot_num=convertFromConstructionSMToSlot(sm,-1);
	
	int ic=(sm-1)*170+i-1;

	//	int ic=(slot_num-1)*1700+(ih4-1);
	
	beta_vec[ic]=beta;
	r25_vec[ic]=r25;
	temp_vec[ic]=1.0;
	nic_vec[ic]=nic;
      }

    fin.close();

    cout << "Writing CaliCrystalIntercalDat objects to database..." << endl;
    cout << "Making a CaliIOV..." << flush;
    CaliIOV caliiov = this->makeCaliIOV();
    cout << "Done." << endl;

    this->printIOV(&caliiov);

    Tm eventTm = caliiov.getSince();
    CaliTag calitag = caliiov.getCaliTag();


    vector<EcalLogicID> ecid_vec;
    //    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_crystal_number", 1,36, 1, 1700,EcalLogicID::NULLID,
    //					   EcalLogicID::NULLID, "EB_T_capsule",12);


    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_T_capsule", 1,36, 1, 170,EcalLogicID::NULLID,
						   EcalLogicID::NULLID, "EB_T_capsule",12);


   map<EcalLogicID, CaliTempDat> dataset;
   int count=0;
   //   for (int ii=0; ii<61200; ii++){
   for (int ii=0; ii<6120; ii++){
       CaliTempDat capTemp;
       if(temp_vec[ii]!=0 ){
       
	 float val1 = beta_vec[ii] ;
	 float val2 = r25_vec[ii];
       
	 capTemp.setBeta(val1);
	 capTemp.setR25(val2);
	 capTemp.setOffset(nic_vec[ii]);
	 capTemp.setTaskStatus(1) ;
       
       // Fill the dataset
	 
	 dataset[ecid_vec[ ii] ] = capTemp;
	 if(ii/100==0) cout<< "filling DB for "<<ii<<" count="<<count<<endl; 
	 count++;
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
