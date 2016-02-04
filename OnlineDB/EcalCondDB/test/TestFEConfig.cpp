#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>
#include <cstdlib>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalsDat.h"
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
  

  int test(int runa, int runb ) {


    cout << "*****************************************" << endl;
    cout << "************Inserting Peds from OMDS*****" << endl;
    cout << "*****************************************" << endl;
    
    int result=0;

    cout << "Retrieving run list from DB ... " << endl;
   


    RunTag  my_runtag;
    LocationDef my_locdef;
    RunTypeDef my_rundef;
    
    my_locdef.setLocation("H4B_07");
    my_rundef.setRunType("PEDESTAL");
    my_runtag.setLocationDef(my_locdef);
    my_runtag.setRunTypeDef(my_rundef);
    my_runtag.setGeneralTag("LOCAL");

    // here we retrieve the Monitoring results

    MonVersionDef monverdef;
    monverdef.setMonitoringVersion("test01");
    MonRunTag montag;
    montag.setMonVersionDef(monverdef);
    montag.setGeneralTag("CMSSW");

   
    MonRunList mon_list;
    mon_list.setMonRunTag(montag);
    mon_list.setRunTag(my_runtag);
    int min_run=runa;
    int n_runs=runb;
    cout<< "min run is:"<<min_run<<endl;
    mon_list=econn->fetchMonRunListLastNRuns(my_runtag, montag, min_run ,n_runs  );

    std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();

    cout <<"number of ped runs is : "<< mon_run_vec.size()<< endl;
    int mon_runs=(int) mon_run_vec.size();
    if(mon_runs>0){
      for(int ii=0; ii<mon_runs; ii++){
	cout << "here is the run number: "<< mon_run_vec[ii].getRunIOV().getRunNumber() << endl;
      }
    }

    int sm_num=0;  

    if(mon_runs>0){

      // for the first run of the list we retrieve the pedestals
      int run=0;
      cout <<" retrieve the data for a given run"<< endl;
      cout << "here is the run number: "<< mon_run_vec[run].getRunIOV().getRunNumber() << endl;
      
      RunIOV runiov_prime = mon_run_vec[run].getRunIOV();

      map<EcalLogicID, MonPedestalsDat> dataset_ped;
      econn->fetchDataSet(&dataset_ped, &mon_run_vec[run]);



      typedef map<EcalLogicID, MonPedestalsDat>::const_iterator CImped;
      EcalLogicID ecid_xt;
      MonPedestalsDat  rd_ped;
      
      sm_num=0;
      float ped_m12[61200];
      float ped_m6[61200];
      float ped_m1[61200];
      for (int i=0; i<61200; i++){
	ped_m12[i]=0;
	ped_m6[i]=0;
	ped_m1[i]=0;
      }


      for (CImped p = dataset_ped.begin(); p != dataset_ped.end(); p++) {
	ecid_xt = p->first;
	rd_ped  = p->second;
	sm_num=ecid_xt.getID1();
	int xt_id=ecid_xt.getID2();
	int xt_num=ecid_xt.getID2()+(sm_num-1)*1700-1;
	ped_m12[xt_num]=rd_ped.getPedMeanG12();
	ped_m6[xt_num]=rd_ped.getPedMeanG6();
	ped_m1[xt_num]=rd_ped.getPedMeanG1();
	if(xt_id==123) cout <<"here is one value for XT: "<< xt_id<<" SM: " << sm_num<< " "<< ped_m12[xt_num] << endl;
      }
      //  here we can do something to the pedestals like checking
      //  that they are correct and that there are no missing channels

      // .....


      // then we ship them to the config DB

      int mon_iov_id=mon_run_vec[run].getID();

      cout << "now create fe record " <<endl;
      FEConfigPedInfo fe_ped_info ;
      fe_ped_info.setIOVId(mon_iov_id);
      fe_ped_info.setConfigTag("from_CondDB");
      econn->insertConfigSet(&fe_ped_info);
      result =fe_ped_info.getID();

      // this is to insert in FE Config DB      
      vector<EcalLogicID> ecid_vec;
      ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);
      cout << "*********got channel numbers   *********" << endl;


      map<EcalLogicID, FEConfigPedDat> dataset;
      for (int ich=0; ich<61200; ich++){
	FEConfigPedDat ped;
	ped.setPedMeanG1(ped_m1[ich]);
	ped.setPedMeanG6(ped_m6[ich]);
	ped.setPedMeanG12(ped_m12[ich]);
	// Fill the dataset
	dataset[ecid_vec[ich]] = ped;
      }
    // Insert the dataset, identifying by iov
      cout << "*********about to insert peds *********" << endl;
      econn->insertDataArraySet(&dataset, &fe_ped_info);
      cout << "*********Done peds            *********" << endl;
    
    }
    
    return result;
  }





  void testRead(int iconf_req ) {
    // now we do something else 
    // this is an example for reading the pedestals 
    // for a given config iconf_req 
 
    cout << "*****************************************" << endl;
    cout << "test readinf fe_ped with id="<<iconf_req  << endl;
    cout << "*****************************************" << endl;
 
    FEConfigPedInfo fe_ped_info;
    fe_ped_info.setId(iconf_req);
    econn->fetchConfigSet(&fe_ped_info);
    map<EcalLogicID, FEConfigPedDat> dataset_ped;
    econn->fetchDataSet(&dataset_ped, &fe_ped_info);

    typedef map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
    EcalLogicID ecid_xt;
    FEConfigPedDat  rd_ped;

      float ped_m12[61200];
      float ped_m6[61200];
      float ped_m1[61200];
      for (int i=0; i<61200; i++){
	ped_m12[i]=0;
	ped_m6[i]=0;
	ped_m1[i]=0;
      }

      for (CIfeped p = dataset_ped.begin(); p != dataset_ped.end(); p++) {
	ecid_xt = p->first;
	rd_ped  = p->second;
	int sm_num=ecid_xt.getID1();
	int xt_num=ecid_xt.getID2();
	ped_m12[xt_num]=rd_ped.getPedMeanG12();
	ped_m6[xt_num]=rd_ped.getPedMeanG6();
	ped_m1[xt_num]=rd_ped.getPedMeanG1();
	if(xt_num==123&&sm_num==10) cout <<"here is one value for XT: "<< xt_num<<" " << ped_m12[xt_num] << endl;

      }

    cout << "*****************************************" << endl;
    cout << "test read done"<<iconf_req  << endl;
    cout << "*****************************************" << endl;

  }



  void testWriteLUT() {
    // now we do something else 
    // this is an example for writing the lut
 

      cout << "*****************************************" << endl;
      cout << "************Inserting LUT************" << endl;
      cout << "*****************************************" << endl;

      FEConfigLUTInfo fe_lut_info ;
      fe_lut_info.setNumberOfGroups(1); // this eventually refers to some other table 
      fe_lut_info.setConfigTag("test");
      econn->insertConfigSet(&fe_lut_info);
            
      Tm tdb = fe_lut_info.getDBTime();
      //tdb.dumpTm();

      vector<EcalLogicID> ecid_vec;
      ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);
     
      map<EcalLogicID, FEConfigLUTGroupDat> dataset;
      // we create 5 LUT groups 
      for (int ich=0; ich<5; ich++){
	FEConfigLUTGroupDat lut;
	lut.setLUTGroupId(ich);
	for (int i=0; i<1024; i++){
	  lut.setLUTValue(i, (i*2) );
	}
	// Fill the dataset
	dataset[ecid_vec[ich]] = lut; // we use any logic id, because it is in any case ignored... 
       }

      // Insert the dataset
      econn->insertDataArraySet(&dataset, &fe_lut_info);



      // now we store in the DB the correspondence btw channels and LUT groups 
      map<EcalLogicID, FEConfigLUTDat> dataset2;
      // in this case I decide in a stupid way which channel belongs to which group 
      for (int ich=0; ich<(int)ecid_vec.size() ; ich++){
	FEConfigLUTDat lut;
	int igroup=ich/(ecid_vec.size()/5);
	lut.setLUTGroupId(igroup);
	// Fill the dataset
	dataset2[ecid_vec[ich]] = lut;
      }

      // Insert the dataset

      econn->insertDataArraySet(&dataset2, &fe_lut_info);
      cout << "*****************************************" << endl;
      cout << "************LUT done*********************" << endl;
      cout << "*****************************************" << endl;

    

  }

  void testWriteWeights() {
    // now we do something else 
    // this is an example for writing the weights
 
      cout << "*****************************************" << endl;
      cout << "************Inserting weights************" << endl;
      cout << "*****************************************" << endl;

      FEConfigWeightInfo fe_wei_info ;
      fe_wei_info.setNumberOfGroups(5); // this eventually refers to some other table 
      fe_wei_info.setConfigTag("my preferred weights");
      econn->insertConfigSet(&fe_wei_info);
      
      Tm tdb = fe_wei_info.getDBTime();
      //      tdb.dumpTm();

      vector<EcalLogicID> ecid_vec;
      ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);

      map<EcalLogicID, FEConfigWeightGroupDat> dataset;
      // we create 5 groups 
      for (int ich=0; ich<5; ich++){
	FEConfigWeightGroupDat wei;
	wei.setWeightGroupId(ich);
	wei.setWeight0(0);
	wei.setWeight1(1);
	wei.setWeight2(3);
	wei.setWeight3(3);
	wei.setWeight4(3);
	// Fill the dataset
	dataset[ecid_vec[ich]] = wei; // we use any logic id, because it is in any case ignored... 
       }

      // Insert the dataset
      econn->insertDataArraySet(&dataset, &fe_wei_info);



      // now we store in the DB the correspondence btw channels and LUT groups 
      map<EcalLogicID, FEConfigWeightDat> dataset2;
      // in this case I decide in a stupid way which channel belongs to which group 
      for (int ich=0; ich<(int)ecid_vec.size() ; ich++){
	FEConfigWeightDat weid;
	int igroup=ich/(ecid_vec.size()/5);
	weid.setWeightGroupId(igroup);
	// Fill the dataset
	dataset2[ecid_vec[ich]] = weid;
      }

      // Insert the dataset

      econn->insertDataArraySet(&dataset2, &fe_wei_info);
      cout << "*****************************************" << endl;
      cout << "************weights done*****************" << endl;
      cout << "*****************************************" << endl;

    

  }



private:
  CondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  TFile* f;
  TH2F* mataq_vs_run;
  TH2F* apd_pn_mean_vs_run;

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
  int n_run=atoi(sn_run.c_str());


  try {
    CondDBApp app( sid, user, pass);

    int i= app.test(min_run, n_run);
    
    app.testRead(i);

    app.testWriteLUT();
    app.testWriteWeights();
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
