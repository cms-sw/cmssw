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

  void testCompare(std::string tag1, int num1 )
  {

    cout << "Reading config "<<num1<<" from database..." << endl;

    ODRunConfigInfo od1;
    od1.setTag(tag1);
    od1.setVersion(num1);


    econn->fetchConfigSet(&od1);

    cout << "step 1 Done." << endl;

    int ecal_id1= od1.getId();
    int last_cycle_id=0;

    // now the sequence
    RunSeqDef runSeqDef;
    
    ODRunConfigSeqInfo seq1;
    seq1.setEcalConfigId(ecal_id1);
    econn->fetchConfigSet(&seq1);
    int seq_id1=seq1.getId();

    int ncycles=2;
    for (int icy=0; icy<ncycles; icy++){

	// here we read the cycle 

      ODEcalCycle ec_cyc1;
      ec_cyc1.setCycleNum(icy );
      ec_cyc1.setSequenceId(seq_id1);
      econn->fetchConfigSet(&ec_cyc1);
      int cyc_id=ec_cyc1.getId();
      ec_cyc1.printout();

      int dcc_id=ec_cyc1.getDCCId();
      if(dcc_id!=0) {

	ODDCCConfig dcc_prime;
	dcc_prime.setId(dcc_id);
	econn->fetchConfigSet(&dcc_prime);
	dcc_prime.printout();
	unsigned char * buffer_dcc=dcc_prime.getDCCClob();
	cout<< "the DCC clob buffer is:"<<endl;
	std::cout<< "DCC CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;
      }

      int ccs_id=ec_cyc1.getCCSId();
      if(ccs_id!=0) {

	ODCCSConfig ccs_prime;
	ccs_prime.setId(ccs_id);
	econn->fetchConfigSet(&ccs_prime);
	ccs_prime.printout();
      }
      int jbh4_id=ec_cyc1.getJBH4Id();
      if(jbh4_id!=0) {

	ODJBH4Config jbh4_prime;
	jbh4_prime.setId(jbh4_id);
	econn->fetchConfigSet(&jbh4_prime);
	jbh4_prime.printout();
      }
      int dcu_id=ec_cyc1.getDCUId();
      if(dcu_id!=0) {

	ODDCUConfig dcu_prime;
	dcu_prime.setId(dcu_id);
	econn->fetchConfigSet(&dcu_prime);
	dcu_prime.printout();
      }

      int las_id=ec_cyc1.getLaserId();
      if(las_id!=0) {
	ODLaserConfig las_prime ;
	las_prime.setId(las_id);
	econn->fetchConfigSet(&las_prime);
	las_prime.printout();
	unsigned char * buffer_dcc=las_prime.getLaserClob();
	cout<< "the Laser clob buffer is:"<<endl;
	std::cout<< "Laser CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;
      }

      int tcc_id=ec_cyc1.getTCCId();
      if(tcc_id!=0) {
	ODTCCConfig tcc_prime ;
	tcc_prime.setId(tcc_id);
	econn->fetchConfigSet(&tcc_prime);
	tcc_prime.printout();
	unsigned char * buffer_dcc=tcc_prime.getTCCClob();
	cout<< "the TCC clob buffer is:"<<endl;
	std::cout<< "TCC CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;
	unsigned char * buffer_lut=tcc_prime.getLUTClob();
	cout<< "the TCC-LUT clob buffer is:"<<endl;
	std::cout<< "TCC-LUT CLOB:"<<buffer_lut<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_lut;
	unsigned char * buffer_slb=tcc_prime.getSLBClob();
	cout<< "the TCC-SLB clob buffer is:"<<endl;
	std::cout<< "TCC-SLB CLOB:"<<buffer_slb<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_slb;
      }
      int ttcci_id=ec_cyc1.getTTCCIId();
      if(ttcci_id!=0) {
	ODTTCciConfig tcc_prime ;
	tcc_prime.setId(ttcci_id);
	econn->fetchConfigSet(&tcc_prime);
	tcc_prime.printout();
	unsigned char * buffer_tcc=tcc_prime.getTTCciClob();
	cout<< "the TTCci clob buffer is:"<<endl;
	std::cout<< "TTCci CLOB:"<<buffer_tcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_tcc;
      }

      int ttcf_id=ec_cyc1.getTTCFId();
      if(ttcf_id!=0) {
	ODTTCFConfig ttcf_prime ;
	ttcf_prime.setId(ttcf_id);
	econn->fetchConfigSet(&ttcf_prime);
	ttcf_prime.printout();
	unsigned char * buffer_ttcf=ttcf_prime.getTTCFClob();
	cout<< "the TTCF clob buffer is:"<<endl;
	std::cout<< "TTCF CLOB:"<<buffer_ttcf<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_ttcf;
      }





      int ltc_id=ec_cyc1.getLTCId();
      if(ltc_id!=0) {
	ODLTCConfig tcc_prime ;
	tcc_prime.setId(ltc_id);
	econn->fetchConfigSet(&tcc_prime);
	tcc_prime.printout();
	unsigned char * buffer_dcc=tcc_prime.getLTCClob();
	cout<< "the TCCci clob buffer is:"<<endl;
	std::cout<< "TCCci CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;

      }

      int lts_id=ec_cyc1.getLTSId();
      if(lts_id!=0) {
	ODLTSConfig lts_prime ;
	lts_prime.setId(lts_id);
	econn->fetchConfigSet(&lts_prime);
	lts_prime.printout();
      }

      int srp_id=ec_cyc1.getSRPId();
      if(srp_id!=0) {
	ODSRPConfig srp_prime ;
	srp_prime.setId(srp_id);
	econn->fetchConfigSet(&srp_prime);
	srp_prime.printout();
	unsigned char * buffer_dcc=srp_prime.getSRPClob();
	cout<< "the SRP clob buffer is:"<<endl;
	std::cout<< "SRP CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;

      }

      int tccee_id=ec_cyc1.getTCCEEId();
      if(tccee_id!=0) {
	ODTCCEEConfig tccee_prime ;
	tccee_prime.setId(tccee_id);
	econn->fetchConfigSet(&tccee_prime);
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	tccee_prime.printout();
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;

	unsigned char * buffer_lut=tccee_prime.getLUTClob();
	cout<< "the TCCEE-LUT clob buffer is:"<<endl;
	std::cout<< "TCCEE-LUT CLOB:"<<buffer_lut<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_lut;

	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;


	unsigned char * buffer_slb=tccee_prime.getSLBClob();
	cout<< "the TCCEE-SLB clob buffer is:"<<endl;
	std::cout<< "TCCEE-SLB CLOB:"<<buffer_slb<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_slb;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;

	unsigned char * buffer_dcc=tccee_prime.getTCCClob();
	cout<< "the TCCEE clob buffer is:"<<endl;
	std::cout<< "TCCEE CLOB:"<<buffer_dcc<< std::endl;
	cout << "Done." << endl << endl << endl;
	delete [] buffer_dcc;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;
	std::cout<<"------------TCCEE--------------"<<std::endl;

      }

    }
    


    /*

    //CCS
    ODCCSConfig ccs ;
    ccs.setConfigTag("CCSConfig_1");
    ccs.setDaccal(1 );
    ccs.setDelay( 2 );
    ccs.setGain( "3" );
    ccs.setMemGain(  "16" );
    ccs.setOffsetHigh( 10 );
    ccs.setOffsetLow(  20 );
    ccs.setOffsetMid(  15 );
    ccs.setTrgMode(      "unknown" );
    ccs.setTrgFilter(    "standard" );
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&ccs);
    int ccs_id=ccs.getId();
    cout << "CCS Config inserted with ID "<< ccs_id<< endl;


    //DCC
    ODDCCConfig dcc ;
    dcc.setConfigTag("DCCConfig_1");
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
    las.setConfigTag("LasConfig_1");
    las.setMatacqMode("whatever");
    las.setChannelMask(3);
    las.setMaxSamplesForDaq("10");
    las.setPedestalFile("thisfile");
    las.setUseBuffer(1);
    las.setPostTrig(0);
    las.setFPMode(1);
    las.setHalModuleFile("bho");
    las.setHalAddressTableFile("bho");
    las.setHalStaticTableFile("bho");
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
    tcc.setConfigTag("TCCConfig_1");
    tcc.setNTestPatternsToLoad(184376);
    std::string my_tccstring="here is a nice DCC configuration";
    char * tcc_clob= new char[my_tccstring.length()];
    strcpy(tcc_clob, my_tccstring.c_str());
    tcc.setTCCClob((unsigned char*)tcc_clob);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&tcc);
    int tcc_id=tcc.getId();
    cout << "TCC Config inserted with ID "<< tcc_id<< endl;
    delete [] tcc_clob;


    // TTCci

    cout << "TTCci now coming "<< endl;

    ODTTCciConfig ttcci ;
    ttcci.setConfigTag("TTCciConfig_1");
    ttcci.setTrgMode("unknown");
    std::string my_stringx="here is a nice TTCci configuration";
    char * ttcci_clob= new char[my_stringx.length()];
    strcpy(ttcci_clob, my_stringx.c_str());
    ttcci.setTTCciClob((unsigned char*)ttcci_clob);
    // write to the DB
    cout << "Inserting TTCci in DB..." << endl;
    econn->insertConfigSet(&ttcci);
    cout << "here we are..." << endl;
    int ttc_id=ttcci.getId();
    cout << "TTCci Config inserted with ID "<< ttc_id<< endl;
    delete [] ttcci_clob;




    // LTC
    ODLTCConfig ltc ;
    ltc.setConfigTag("LTCConfig_1");
    ltc.setLTCConfigurationFile("a_file.xml");
    std::string my_stringltc="here is a nice ltC configuration";
    char * ltc_clob= new char[my_stringltc.length()];
    strcpy(ltc_clob, my_stringltc.c_str());
    ltc.setLTCClob((unsigned char*)ltc_clob);
    // write to the DB
    cout << "Inserting in DB..." << endl;
    econn->insertConfigSet(&ltc);
    int ltc_id=ltc.getId();
    cout << "LTC Config inserted with ID "<< ltc_id<< endl;
    delete [] ltc_clob;

    // LTS
    ODLTSConfig lts ;
    lts.setConfigTag("LTSConfig_1");
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
    jbh4.setConfigTag("JBH4Config_1");
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
    cout << "Inserting in DB the ODRunConfigInfo..." << flush;
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

	// here we insert the cycle 

	ODEcalCycle ec_cyc;
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
	ec_cyc.setJBH4Id( jbh4_id       );
	ec_cyc.setScanId( 0       );
	ec_cyc.setSequenceId(seq_id);


	econn->insertConfigSet(&ec_cyc);

	int cyc_id=ec_cyc.getId();
	last_cycle_id=cyc_id;
	cout << "Last Cycle inserted with ID "<< cyc_id<< endl;
      

      }
    }





    */

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

  string tag1;


  int num1;


  if (argc != 7) {
    cout << "Usage:" << endl;
    cout << "  " << argv[0] << " <host> <SID> <user> <pass> <tag1> <version1> " << endl;
    exit(-1);
  }

  host = argv[1];
  sid = argv[2];
  user = argv[3];
  pass = argv[4];
  tag1 = argv[5];
  num1 = atoi(argv[6]);
  //  tag2 = argv[7];
  // num2 = atoi(argv[8]);

  try {
    CondDBApp app(host, sid, user, pass);

    app.testCompare(tag1, num1);

  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
