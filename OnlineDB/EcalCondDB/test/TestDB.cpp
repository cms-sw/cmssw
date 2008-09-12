#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TH2F.h"
#include "TF1.h"


using namespace std;

class CondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  CondDBApp(string host, string sid, string user, string pass, int port)
  {
    try {
      cout << "Making connection...to " << port << flush;
      econn = new EcalCondDBInterface( host, sid, user, pass, port );
      cout << "Done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
      exit(-1);
    }

 
  }
  CondDBApp(string sid, string user, string pass)
  {
    try {
      cout << "Making connection...to " << sid << endl;
      econn = new EcalCondDBInterface(  sid, user, pass );
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

inline std::string to_string( char value[])
                                                                                
{
                                                                                
    std::ostringstream streamOut;
                                                                                
    streamOut << value;
                                                                                
    return streamOut.str();
                                                                                
}
                                                                                
                                                                                

  RunIOV makeRunIOV(int run_num, std::string& location)
  {
    cout << "entering makeRunIOV for run " << run_num<< " and location= "<< location << endl;
    // The objects necessary to identify a dataset

    LocationDef locdef;
    RunTypeDef rundef;
    RunTag runtag;

    locdef.setLocation(location);
 
    rundef.setRunType("LASER");
     
    runtag.setLocationDef(locdef);
    runtag.setRunTypeDef(rundef);
    runtag.setGeneralTag("LASER"); 
    cout << " set the run tag and location " << endl;

    // Our beginning time will be the current GMT time
    // This is the time zone that should be written to the DB!
    // (Actually UTC)
    Tm startTm;
    startTm.setToCurrentGMTime();

    uint64_t microseconds = startTm.microsTime();
    startmicros = microseconds;


    cout << "Starting Time:    " << startTm.str() << endl;
    cout << "Starting Micros:  " << startmicros << endl;

    // Set the properties of the iov
    RunIOV runiov;

    startTm.setToMicrosTime(microseconds);
    cout << "Setting run " << run_num << " run_start " << startTm.str() << endl;
    runiov.setRunNumber(run_num);
    runiov.setRunStart(startTm);
    runiov.setRunTag(runtag);
    
    return runiov;

  }

  
  LMFRunIOV makeLMFRunIOV(RunIOV* runiov)
  {
    // LMF Tag and IOV
    
    LMFRunTag lmftag;
    
    LMFRunIOV lmfiov;
    lmfiov.setLMFRunTag(lmftag);
    lmfiov.setRunIOV(*runiov);
    lmfiov.setSubRunNumber(0);
    lmfiov.setSubRunStart(runiov->getRunStart());

    return lmfiov;
  }



  /**
   *  Write LMFLaserBlueRawDat objects with associated RunTag and RunIOVs
   *  IOVs are written using automatic updating of the 'RunEnd', as if
   *  the time of the end of the run was not known at the time of writing.
   */
  void testWrite(int run_num)
  {
    std::string location="H4B"; // this is H4 beam 
    cout << "Writing LMFLaserBlueDat to database..." << endl;
    RunIOV runiov = this->makeRunIOV(run_num, location);
    RunTag runtag = runiov.getRunTag();
    run_t run = runiov.getRunNumber();

    // write to the DB
    cout << "Inserting run..." << flush;
    econn->insertRunIOV(&runiov);
    cout << "Done inserting run" << endl;


    cout << "Fetching run by tag just used..." << flush;
    RunIOV runiov_prime = econn->fetchRunIOV(&runtag, run);

    // LMF Tag and IOV
    LMFRunIOV lmfiov = this->makeLMFRunIOV(&runiov_prime);
    map<EcalLogicID, LMFLaserBlueNormDat> dataset_lmf;
    map<EcalLogicID, LMFLaserBlueRawDat> dataset_lraw;
    map<EcalLogicID, LMFPNBlueDat> dataset_lpn;
    map<EcalLogicID, LMFMatacqBlueDat> dataset_mq;
    
    vector<EcalLogicID> ecid_vec;
    int sm_min=1;
    int sm_max=36;
    int ch_min=1;
    int ch_max=1700;


    Tm startTm3;
    startTm3.setToCurrentGMTime();
    cout << "query starting at:    " << startTm3.str() << endl;

    ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", sm_min, sm_max, ch_min, ch_max);

    Tm startTm4;
    startTm4.setToCurrentGMTime();
    cout << "query finished at:    " << startTm4.str() << endl;


    vector<EcalLogicID> ecid_vec_pn;
    ch_min=0;
    ch_max=9;
    ecid_vec_pn = econn->getEcalLogicIDSet("EB_LM_PN", sm_min, sm_max,ch_min, ch_max);

    Tm startTm5;
    startTm5.setToCurrentGMTime();
    cout << "query PN finished at:    " << startTm5.str() << endl;

    int count=0;
    int count_pn=0;
    for (int sm_num=1; sm_num<37; sm_num++){


    // Get channel ID for SM sm_num, crystal c      
      
      Int_t cx, cy;

      int pn_chan[]={1,101, 501, 901, 1301, 11, 111, 511, 911, 1311 };

      for (int c=1; c<1701; c++){
	// the channels are turned in phi and eta 
	// with respect to the standard numbering scheme
	cx = 85-(c-1)/20;
	cy = 20-(c-1)%20;
	float x= rand()/1e8;
	float apdpn = 1.5+x;
	float apd = 1500.+x;
	float apd_rms = 15.5+x;
	float apdpn_rms = 0.1+x;
	float pn = 800.+x;
	
	if(c%500==0) cout << "SM: "<<sm_num<<  " channel "<< c<< " value "<< apdpn << endl; 


	int ispn=-1;
	for(int j=0; j<10; j++){
	  if(pn_chan[j]==c) ispn=j;
	}
	
	if(ispn!=-1){
	  LMFPNBlueDat bluepn;
	  
	  bluepn.setPNPeak(pn);
	  bluepn.setPNErr(pn/100.);
	  dataset_lpn[ecid_vec_pn[count_pn]] = bluepn;
	  count_pn++;
	}
	
	// Set the data
	LMFLaserBlueNormDat bluelaser;
	LMFLaserBlueRawDat bluelaserraw;
	bluelaser.setAPDOverPNMean(apdpn);
	bluelaser.setAPDOverPNAMean(apdpn);
	bluelaser.setAPDOverPNBMean(apdpn);
	bluelaser.setAPDOverPNARMS(apdpn_rms);
	bluelaser.setAPDOverPNBRMS(apdpn_rms);
	bluelaser.setAPDOverPNRMS(apdpn_rms);
	
	bluelaserraw.setAPDPeak(apd);
	bluelaserraw.setAPDErr(apd_rms);
	
	// Fill the dataset
	dataset_lmf[ecid_vec[count]] = bluelaser;
	dataset_lraw[ecid_vec[count]] = bluelaserraw;

	count++;

	
      }
    

    }
  
    cout << "finished processing the APD and PN data"  << endl;
    
    cout << "now MATACQ  "<<  endl;
    
    try {
      
	float x=rand()/1e8;

	float par_height= 100.+x;
	
	float par_width=10.+x;
	
	float par_timing=10.+x;
	
	LMFMatacqBlueDat lmf_mq;
	EcalLogicID ecid_mq;
	ecid_mq = econn->getEcalLogicID("EB");
	lmf_mq.setAmplitude(par_height);
	lmf_mq.setWidth(par_width);
	lmf_mq.setTimeOffset(par_timing);
	dataset_mq[ecid_mq] = lmf_mq;
	econn->insertDataSet(&dataset_mq, &lmfiov);
      
	cout << "Done Matacq "  << endl;
    
    } catch (...) { 
      cout << "TestDB>> error with MATACQ " << endl ;
    } 

    Tm startTm;
    startTm.setToCurrentGMTime();
    cout << "record generated at:    " << startTm.str() << endl;


    // Insert the dataset, identifying by iov
    cout << "Inserting dataset..." << flush;
    econn->insertDataArraySet(&dataset_lmf, &lmfiov);
    econn->insertDataArraySet(&dataset_lraw, &lmfiov);
    econn->insertDataArraySet(&dataset_lpn, &lmfiov);
    cout << "Done." << endl;

    Tm startTm2;
    startTm2.setToCurrentGMTime();
    cout << "record inserted at:    " << startTm2.str() << endl;

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
  void printDataSet( const map<EcalLogicID, LMFLaserBlueRawDat>* dataset, int limit = 0 ) const
  {
    cout << "==========printDataSet()" << endl;
    if (dataset->size() == 0) {
      cout << "No data in map!" << endl;
    }
    EcalLogicID ecid;
    LMFLaserBlueRawDat bluelaser;

    int count = 0;
    typedef map< EcalLogicID, LMFLaserBlueRawDat >::const_iterator CI;
    for (CI p = dataset->begin(); p != dataset->end(); p++) {
      count++;
      if (limit && count > limit) { return; }
      ecid = p->first;
      bluelaser  = p->second;
     
      cout << "SM:                     " << ecid.getID1() << endl;
      cout << "Xtal:                   " << ecid.getID2() << endl;
      cout << "APD Peak:               " << bluelaser.getAPDPeak() << endl;
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
  string sport;
  string smin_run;
  string sn_run;

  if (argc != 8) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <port> <min_run> <n_run>" << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  sport = argv[5];
  int port=atoi(sport.c_str());
  smin_run = argv[6];
  int min_run=atoi(smin_run.c_str());
  sn_run = argv[7];
  int n_run=atoi(sn_run.c_str())+min_run;

  try {
    //    CondDBApp app(host, sid, user, pass, port);
    CondDBApp app( sid, user, pass);

    for (int i=min_run; i<n_run; i++){
      int run_num=i;
      cout << "going to write run " <<run_num << endl;
      app.testWrite(run_num);
    }


  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
