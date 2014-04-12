#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"

#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunLaserRunDat.h"

// fixme
#include "OnlineDB/EcalCondDB/interface/LMFLaserPrimDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFLaserPNPrimDat.h"
#include "OnlineDB/EcalCondDB/interface/LMFLaserPulseDat.h"

#include "MELaserPrim.hh"


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

  
  LMFRunIOV makeLMFRunIOV(RunIOV* runiov, int subr)
  {
    // LMF Tag and IOV
    
    LMFRunTag lmftag;
    
    LMFRunIOV lmfiov;
    lmfiov.setLMFRunTag(lmftag);
    lmfiov.setRunIOV(*runiov);
    lmfiov.setSubRunNumber(subr);
    lmfiov.setSubRunStart(runiov->getRunStart());
    lmfiov.setSubRunType("Standard");

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
    cout << "Done fetching run" << endl;



    EcalLogicID ecid_allEcal;
    ecid_allEcal = econn->getEcalLogicID("ECAL");
    map< EcalLogicID, RunLaserRunDat >    dataset;
    RunLaserRunDat rd;
    rd.setLaserSequenceType("STANDARD");
    rd.setLaserSequenceCond("STANDALONE");
    // or eventually this:
    // rd.setSequenceType("IN_THE_GAP");
    dataset[ecid_allEcal]=rd;
    econn->insertDataSet( &dataset, &runiov );

    cout << "Done inserting laser config" << endl;

    vector<EcalLogicID> ecid_vec;
    int sm_min=1;
    int sm_max=36;
    int ch_min=0;
    int ch_max=1;
    ecid_vec = econn->getEcalLogicIDSetOrdered("EB_LM_side", sm_min, sm_max, 
					       ch_min, ch_max, 
					       EcalLogicID::NULLID, EcalLogicID::NULLID, 
					       "EB_crystal_number", 1234 );

    vector<EcalLogicID> ecid_vec_pn;
    ch_min=0;
    ch_max=9;
    ecid_vec_pn = econn->getEcalLogicIDSet("EB_LM_PN", sm_min, sm_max,ch_min, ch_max);

    // fanout 
    vector<EcalLogicID> ecid_vec_fanout;
    ch_min=0;
    ch_max=1;
    ecid_vec_fanout = econn->getEcalLogicIDSet("EB_LM_side", sm_min, sm_max,ch_min, ch_max);

    cout << "Done retrieval of logicid " << endl;


    // LMF Tag and IOV
    int subrunnumber = 0;


    for( int color=0; color<1; color++ ) {
	for (int sm_num=1; sm_num<37; sm_num++){
	  for( int side=0; side<2; side++ ) {

            subrunnumber ++;

	    cout << "going to generate lmf run iov ..." << endl;
            LMFRunIOV lmfiov = this->makeLMFRunIOV( &runiov, subrunnumber );
	    cout << "Done generating lmf run iov " << endl;

            // datasets
            //      typedef LMFPNBluePrimDat LMFLaserBluePNPrimDat;  // name of class should be fixed
            map< EcalLogicID, LMFRunDat             >    dataset_lmfrun;
            map< EcalLogicID, LMFLaserConfigDat     >    dataset_config;
            //      map< EcalLogicID, LMFLaserBluePrimDat   >      dataset_prim;
            map< EcalLogicID, LMFLaserPrimDat   >          dataset_prim;
            //      map< EcalLogicID, LMFLaserBluePNPrimDat >    dataset_pnprim;
            map< EcalLogicID, LMFLaserPNPrimDat >        dataset_pnprim;
            //      map< EcalLogicID, LMFLaserBluePulseDat  >     dataset_pulse;
            map< EcalLogicID, LMFLaserPulseDat  >     dataset_pulse;

            // Set the data

            LMFRunDat lmf_lmfrun;
            lmf_lmfrun.setNumEvents(  150  );
            lmf_lmfrun.setQualityFlag(   1 );
	    int idchannel=(sm_num-1)*2+side; 
	    std::cout << "channel "<< idchannel << endl;

	    dataset_lmfrun[ ecid_vec_fanout[(sm_num-1)*2+side] ] = lmf_lmfrun;
            cout << "Inserting lmf run..." << flush;
            econn->insertDataSet( &dataset_lmfrun, &lmfiov );
            cout << "Done." << endl;


            LMFLaserConfigDat lmf_config;
            lmf_config.setWavelength( 492     );
            lmf_config.setVFEGain(    12       );
            lmf_config.setPNGain(     16        );
            lmf_config.setPower(      100     );
            lmf_config.setAttenuator( 1 );
            lmf_config.setCurrent(    100    );
            lmf_config.setDelay1(     0    );
            lmf_config.setDelay2(     0    );

	    dataset_config[ecid_vec_fanout[(sm_num-1)*2+side]] = lmf_config;
            cout << "Inserting lmf config..." << flush;
            econn->insertDataSet( &dataset_config, &lmfiov );
            cout << "Done." << endl;

            //
            // Laser MATACQ Primitives
            //
	    LMFLaserPulseDat::setColor( color ); // set the color
            LMFLaserPulseDat lmf_pulse;
            lmf_pulse.setFitMethod( 0 );  // fixme -- is it a string or an int ???
            lmf_pulse.setAmplitude( 100.4   );
            lmf_pulse.setTime(     34.5   );
            lmf_pulse.setRise(     3.2  );
            lmf_pulse.setFWHM(     33.6  );
            lmf_pulse.setFW80(    44.5    );
            lmf_pulse.setFW20(     45.3   );
            lmf_pulse.setSliding(  43.2 );
            // Fill the dataset
            dataset_pulse[ecid_vec_fanout[(sm_num-1)*2+side]] = lmf_pulse;
            cout << "Inserting lmf pulse ..." << flush;
            econn->insertDataSet( &dataset_pulse, &lmfiov );
            cout << "Done." << endl;


	    // LASER BLUE PRIM Data 
	    LMFLaserPrimDat::setColor( color ); // set the color
	    int nchan=0;
	    if(side==0) nchan=800;
	    if(side==1) nchan=900;

            for( int ixt=1; ixt<=nchan;  ixt++ ) {


	      float x= rand()/1e8;
	      float apdpn = 1.5+x;
	      float apd = 1500.+x;
	      float apd_rms = 15.5+x;
	      float apdpn_rms = 0.1+x;
	    

	      EcalLogicID ecid_prim = ecid_vec[(sm_num-1)*1700+side*800+ixt -1]; 
	      // Set the data
	      LMFLaserPrimDat bluelaser;
	      bluelaser.setFlag(   1 );
	      bluelaser.setMean(apd  );
	      bluelaser.setRMS( apd_rms  );
	      bluelaser.setPeak(apd  );
	      bluelaser.setAPDOverPNAMean(apdpn );
	      bluelaser.setAPDOverPNARMS(apdpn_rms  );
	      bluelaser.setAPDOverPNAPeak( apdpn );  
	      bluelaser.setAPDOverPNBMean(apdpn);
	      bluelaser.setAPDOverPNBRMS( apdpn_rms  );
	      bluelaser.setAPDOverPNBPeak(apdpn );
	      
	      bluelaser.setAPDOverPNMean( apdpn );
	      bluelaser.setAPDOverPNRMS( apdpn_rms  );
	      bluelaser.setAPDOverPNPeak(apdpn  );
	      
	      bluelaser.setAlpha(   1.2       );
	      bluelaser.setBeta(    1.3       );
	      bluelaser.setShapeCor( 100.2      );
	      // Fill the dataset
	      dataset_prim[ecid_prim] = bluelaser;
	  
	  }


            // Inserting the dataset, identified by iov
            cout << "Inserting  _PRIM_DAT  ..." << flush;
            econn->insertDataSet( &dataset_prim,   &lmfiov );
            cout << "Done." << endl;

	    LMFLaserPNPrimDat::setColor( color ); // set the color

	    nchan=10;
            for( int ipn=1; ipn<=nchan ; ipn++ ) {

	      float x= rand()/1e8;
	      float pn = 800.+x;

	      EcalLogicID ecid_pn = ecid_vec_pn[(sm_num-1)*10+ipn -1];
	      // Set the data
	      LMFLaserPNPrimDat bluepn;
	      bluepn.setFlag(1);
	      bluepn.setMean( pn );
	      bluepn.setRMS( pn/10. );
	      bluepn.setPeak( pn);
	      bluepn.setPNAOverPNBMean( 1.+x );
	      bluepn.setPNAOverPNBRMS((1+x)*0.1 );
	      bluepn.setPNAOverPNBPeak( 1.+x);
	      // Fill the dataset
	      dataset_pnprim[ecid_pn] = bluepn;
	    }

            cout << "Inserting _PN_PRIM_DAT ..." << flush;
            econn->insertDataSet( &dataset_pnprim, &lmfiov );
            cout << "Done." << endl;

	  }
	}
      }
  
    cout << "finished processing the APD and PN data"  << endl;
    

    

    Tm startTm;
    startTm.setToCurrentGMTime();
    cout << "program finished at:    " << startTm.str() << endl;

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
  string smin_run;
  string sn_run;

  if (argc != 6) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <SID> <user> <pass> <min_run> <n_run>" << endl;
    exit(-1);
  }

  sid = argv[1];
  user = argv[2];
  pass = argv[3];
  smin_run = argv[4];
  int min_run=atoi(smin_run.c_str());
  sn_run = argv[5];
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
