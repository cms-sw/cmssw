#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <typeinfo>
#include <sstream>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_cali_types.h"


using std::string;

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


    int slot_num=0;


    FILE *inpFile; // input file

    inpFile = fopen(filename.c_str(),"r");
    if(!inpFile) {
      std::cout<< "*** Can not open file: "<<filename;
    
    }

    char line[256];

    std::ostringstream str;

    fgets(line,255,inpFile);
    string sm_or_all=to_string(line);
    int sm_number=0;
    int nchan=1700;
    sm_number=atoi(line);
    str << "sm= " << sm_number << endl ;
    if(sm_number!=-1){
      nchan=1700;
    } else {
      nchan=61200;
    }
    fgets(line,255,inpFile);
    //int nevents=atoi(line); // not necessary here just for online conddb

    fgets(line,255,inpFile);
    string gen_tag=to_string(line);
    str << "gen tag " << gen_tag << endl ;  // should I use this?

    fgets(line,255,inpFile);
    string cali_method=to_string(line);
    str << "cali method " << cali_method << endl ; // not important

    fgets(line,255,inpFile);
    string cali_version=to_string(line);
    str << "cali version " << cali_version << endl ; // not important

    fgets(line,255,inpFile);
    string cali_type=to_string(line);
    str << "cali type " << cali_type << endl ; // not important

    std::cout << "Intercalibration file " << str.str() ;


    LocationDef locdef;
    locdef.setLocation("LAB");
    CaliTag calitag;
    calitag.setLocationDef(locdef);
    calitag.setMethod(cali_method);
    calitag.setVersion(cali_version);
    calitag.setDataType(cali_type);
    calitag.setGeneralTag(gen_tag);


    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm since ;
    since.setToGMTime(1);   

    // Set the properties of the iov
    CaliIOV caliiov;

    caliiov.setSince(since);
    caliiov.setCaliTag(calitag);


    int sm_num[61200]={0};
    int cry_num[61200]={0};
    float calib[61200]={0};
    float calib_rms[61200]={0};
    int calib_nevents[61200]={0};
    int calib_status[61200]={0};


    vector<EcalLogicID> ecid_vec;
    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_crystal_number", 1, 36, 1, 1700,EcalLogicID::NULLID,
						   EcalLogicID::NULLID, "EB_crystal_number",12);


    int ii = 0;
    if(sm_number!=-1){
      while(fgets(line,255,inpFile)) {
        sscanf(line, "%d %f %f %d %d", &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_nevents[ii], &calib_status[ii]);
        sm_num[ii]=sm_number;
        ii++ ;
      }
    } else {
      // this is for the whole Barrel
      cout<<"mode ALL BARREL" <<endl;
      while(fgets(line,255,inpFile)) {
        sscanf(line, "%d %d %f %f %d", &sm_num[ii], &cry_num[ii], &calib[ii], &calib_rms[ii], &calib_status[ii] );
        if(ii==0) cout<<"crystal "<<cry_num[ii]<<" of sm "<<sm_num[ii]<<" cali= "<< calib[ii]<<endl;
        ii++ ;
      }
    }

    fclose(inpFile);           // close inp. file

    cout << " I read the calibrations for "<< ii<< " crystals " << endl;
    if(ii!=nchan) std::cout << "Some crystals missing. Missing channels will be set to 0" << endl;

    // Get channel ID




    cout << "Writing CaliCrystalIntercalDat objects to database..." << endl;

    this->printIOV(&caliiov);

    map<EcalLogicID, CaliCrystalIntercalDat> dataset;

    // Set the data
    for(int i=0; i<nchan; i++){

      slot_num=convertFromConstructionSMToSlot(sm_num[i],-1);
      
      if(i==0) cout<<"crystal "<<cry_num[i]<<" of sm "<<sm_num[i]<< " in slot " <<slot_num<<" calib= "<< calib[i]<<endl	 ;

    CaliCrystalIntercalDat crystalCal;
    crystalCal.setCali(calib[i]);
    crystalCal.setCaliRMS(calib_rms[i]);
    crystalCal.setNumEvents(calib_nevents[i]);
    crystalCal.setTaskStatus(calib_status[i]);

    // Fill the dataset

    int itot=(slot_num-1)*1700+(cry_num[i]-1);
    dataset[ecid_vec [itot] ] = crystalCal;
       
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






private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  int tablesTried;
  int tablesOK;

  std::string to_string( char value[]) {
    std::ostringstream streamOut;
    streamOut << value;
    return streamOut.str();
  }


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

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
