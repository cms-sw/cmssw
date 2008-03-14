#include <iostream>
#include <string>
#include <vector>
#include <time.h>

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

  void testWrite()
  {

    cout << "Writing a config to database..." << endl;
 
    //CCS
    ODCCSConfig ccs ;
    ccs.setDaccal(1 );
    ccs.setDelay( 2 );
    ccs.setGain( 3 );
    ccs.setMemGain(  16 );
    ccs.setOffsetHigh( 10 );
    ccs.setOffsetLow(  20 );
    ccs.setOffsetMid(  15 );
    ccs.setPedestalOffsetRelease( "13-feb-2007" );
    ccs.setSystem(      "my_system" );
    ccs.setTrgMode(      "unknown" );
    ccs.setTrgFilter(    "standard" );
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&ccs);
    int ccs_id=ccs.getId();
    cout << "CCS Config inserted with ID "<< ccs_id<< endl;


    //DCC
    ODDCCConfig dcc ;
    std::string my_string="here is a nice DCC configuration";
    char * dcc_clob= new char[my_string.length()];
    strcpy(dcc_clob, my_string.c_str());
    dcc.setDCCClob((unsigned char*)dcc_clob);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&dcc);
    int dcc_id=dcc.getId();
    cout << "DCC Config inserted with ID "<< dcc_id<< endl;
    delete [] dcc_clob;


    //Laser
    ODLaserConfig las ;
    las.setWaveLength(1 );
    las.setPower( 2 );
    las.setOpticalSwitch( 0 );
    las.setFilter( 1 );
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&las);
    int las_id=las.getId();
    cout << "LASER Config inserted with ID "<< las_id<< endl;


    // TCC
    ODTCCConfig tcc ;
    tcc.setDeviceConfigParamId(184376);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&tcc);
    int tcc_id=tcc.getId();
    cout << "TCC Config inserted with ID "<< tcc_id<< endl;



    // TTCci

    ODTTCciConfig x ;
    std::string my_stringx="here is a nice TTC configuration";
    char * ttc_clob= new char[my_stringx.length()];
    strcpy(ttc_clob, my_stringx.c_str());
    x.setTTCciClob((unsigned char*)ttc_clob);
    // write to the DB
    cout << "Inserting TTCci in DB..." << endl;
    econn->insertConfigSet(&x);
    cout << "here we are..." << endl;
    int ttc_id=x.getId();
    cout << "TTCci Config inserted with ID "<< ttc_id<< endl;
    delete [] ttc_clob;

    // mataq
    ODMataqConfig maq ;
    maq.setMataqMode("whatever");
    maq.setFastPedestal(1);
    maq.setChannelMask(3);
    maq.setMaxSamplesForDaq("10");
    maq.setPedestalFile("thisfile");
    maq.setUseBuffer(1);
    maq.setPostTrig(0);
    maq.setFPMode(1);
    maq.setHalModuleFile("bho");
    maq.setHalAddressTableFile("bho");
    maq.setHalStaticTableFile("bho");
    maq.setMataqSerialNumber("9287658346");
    maq.setPedestalRunEventCount(1000);
    maq.setRawDataMode(0);
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&maq);
    int maq_id=maq.getId();
    cout << "MATAQ Config inserted with ID "<< maq_id<< endl;



    // LTC
    ODLTCConfig ltc ;
    ltc.setDeviceConfigParamId(184376);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&ltc);
    int ltc_id=ltc.getId();
    cout << "LTC Config inserted with ID "<< ltc_id<< endl;

    // LTS
    ODLTSConfig lts ;
    lts.setTriggerType("trigger_type");
    lts.setNumberOfEvents(2147);
    lts.setRate(200);
    lts.setTrigLocL1Delay(10);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&lts);
    int lts_id=lts.getId();
    cout << "LTs Config inserted with ID "<< lts_id<< endl;

    // JBH4
    ODJBH4Config jbh4 ;
    jbh4.setUseBuffer(1);
    jbh4.setHalModuleFile("bho");
    jbh4.setHalAddressTableFile("bho");
    jbh4.setHalStaticTableFile("bho");
    jbh4.setCbd8210SerialNumber("9287658346");
    jbh4.setCaenBridgeType("CAEN");
    jbh4.setCaenLinkNumber(2);
    jbh4.setCaenBoardNumber(34);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&jbh4);
    int jbh4_id=jbh4.getId();
    cout << "JBH4 Config inserted with ID "<< jbh4_id<< endl;


    // now the ECAL_RUN_CONFIGURATION 
    RunTypeDef rundef;
    rundef.setRunType("TEST");
    RunModeDef runmode;
    runmode.setRunMode("LOCAL");

    int nseq=3;

    ODRunConfigInfo run_cfg;
    run_cfg.setId(1);
    run_cfg.setTag("test");
    run_cfg.setDescription("this is a test");
    run_cfg.setVersion(1);
    run_cfg.setNumberOfSequences(nseq);
    run_cfg.setRunTypeDef(rundef);
    run_cfg.setRunModeDef(runmode);
    cout << "Inserting in DB..." << flush;
    econn->insertConfigSet(&run_cfg);
    cout << "Done." << endl;
    int ecal_id= run_cfg.getId();
    int last_cycle_id=0;

    // now the sequences
    for (int iseq=0; iseq<nseq; iseq++){
      RunSeqDef runSeqDef;
      runSeqDef.setRunTypeDef(rundef);
      runSeqDef.setRunSeq("TEST_SEQUENCE");

      int ncycles=2;
      ODRunConfigSeqInfo seq;
      seq.setEcalConfigId(ecal_id);
      seq.setDescription("this is a seq");
      seq.setNumberOfCycles(ncycles);
      seq.setSequenceNumber(iseq);
      seq.setRunSeqDef(runSeqDef);
      // here we insert the sequence and get back the id 
      cout << "Inserting in DB..." << flush;
      econn->insertConfigSet(&seq);
      int seq_id=seq.getId();
      cout << "Seq inserted with ID "<< seq_id<< endl;

      for (int icy=0; icy<ncycles; icy++){
	/*
	// here we insert the cycle 
	ODRunConfigCycleInfo cyc;
	cyc.setTag("cycle_tag");
	cyc.setDescription("the beautiful cycle");
	cyc.setCycleNumber(icy);
	cyc.setSequenceID(seq_id);
	// here we insert the cycle and get back the id 
	cout << "Inserting cycle " << icy << " in DB..." << flush;
	econn->insertConfigSet(&cyc);
	int cyc_id=cyc.getId();
	cout << "Cycle inserted with ID "<< cyc_id<< endl;

	// then for each object that partecipates in the cycle we 
	// insert one entry in the cycle-object table that points to the actual config 

	ODCCSCycle ccs_cycle;
	ccs_cycle.setId(cyc_id);
	ccs_cycle.setCCSConfigurationID(ccs_id);
	econn->insertConfigSet(&ccs_cycle);
	cout << "Inserting CCS-cycle in DB..." << flush;

	ODDCCCycle dcc_cycle;
	dcc_cycle.setId(cyc_id);
	dcc_cycle.setDCCConfigurationID(dcc_id);
	econn->insertConfigSet(&dcc_cycle);
	cout << "Inserting DCC-cycle in DB..." << flush;

	ODLaserCycle las_cycle;
	las_cycle.setId(las_id);
	las_cycle.setLaserConfigurationID(las_id);
	econn->insertConfigSet(&las_cycle);
	cout << "Inserting DCC-cycle in DB..." << flush;

	// or just define ODEcalCycle and write 

	*/

	ODEcalCycle ec_cyc;
	//	ec_cyc.setId(cyc_id);
	// we don't need the following 
        //  to write the ecal cycle
	//	ec_cyc.setTag( );
	//	ec_cyc.setVersion( );
	//	ec_cyc.setSeqNum(  );
	ec_cyc.setCycleNum(icy );
	ec_cyc.setCycleTag("cycle_tag" );
	ec_cyc.setCycleDescription("the beautiful cycle" );
	ec_cyc.setCCSId(  ccs_id       );
	ec_cyc.setDCCId(  dcc_id       );
	ec_cyc.setLaserId(las_id       );
	ec_cyc.setLTCId(  ltc_id       );
	ec_cyc.setLTSId(  lts_id       );
	ec_cyc.setTCCId(  tcc_id       );
	ec_cyc.setTTCCIId(ttc_id       );
	ec_cyc.setMataqId(maq_id       );
	ec_cyc.setJBH4Id( jbh4_id       );
	ec_cyc.setScanId( 0       );
	ec_cyc.setSequenceId(seq_id);


	econn->insertConfigSet(&ec_cyc);

	int cyc_id=ec_cyc.getId();
	last_cycle_id=cyc_id;
	cout << "Last Cycle inserted with ID "<< cyc_id<< endl;
      

      }
    }

    ODEcalCycle ecal_cycle; 
    ecal_cycle.setId(last_cycle_id);
    econn->fetchConfigSet(&ecal_cycle);
    ecal_cycle.printout();
    int dcc_prime_id=ecal_cycle.getDCCId();
    ODDCCConfig dcc_prime;
    dcc_prime.setId(dcc_prime_id);
    econn->fetchConfigSet(&dcc_prime);
    unsigned char * buffer_dcc=dcc_prime.getDCCClob();
    cout<< "the DCC clob buffer is:"<<endl;
    std::cout<< "DCC CLOB:"<<buffer_dcc<< std::endl;
    cout << "Done." << endl << endl << endl;
    delete [] buffer_dcc;
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
    CondDBApp app(host, sid, user, pass);

    app.testWrite();

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
