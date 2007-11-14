#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

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
  

  void test(int runa, int runb ) {

    cout << "Retrieving run list from DB ... " << endl;
    RunList my_runlist ;
    RunTag  my_runtag;
    LocationDef my_locdef;
    RunTypeDef my_rundef;
    
    my_locdef.setLocation("H4B_07");
    my_rundef.setRunType("PEDESTAL");
    my_runtag.setLocationDef(my_locdef);
    my_runtag.setRunTypeDef(my_rundef);
    my_runtag.setGeneralTag("LOCAL");

    my_runlist=econn->fetchRunList(my_runtag);

    std::vector<RunIOV> run_vec=  my_runlist.getRuns();

    float day_to_sec=24.*60.*60.;
    float two_hours=2.*60.*60.;
    
    
    cout <<"number of runs is : "<< run_vec.size()<< endl;
    int nruns=run_vec.size();
    if(nruns>0){
      
      cout << "here is first run : "<< run_vec[0].getRunNumber() << endl;
      // cout << "here is the run Start for first run ="<< run_vec[0].getRunStart()<< endl;
      // cout << "here is the run End for first run =" <<run_vec[0].getRunEnd()<< endl;

    }

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
    int max_run=runb;
    mon_list=econn->fetchMonRunList(my_runtag, montag,min_run,max_run );
    std::vector<MonRunIOV> mon_run_vec=  mon_list.getRuns();

    cout <<"number of mon runs is : "<< mon_run_vec.size()<< endl;
    int mon_runs=mon_run_vec.size();
    if(mon_runs>0){
      cout << "here is first sub run : "<< mon_run_vec[0].getSubRunNumber() << endl;
      cout << "here is the run number: "<< mon_run_vec[0].getRunIOV().getRunNumber() << endl;
    }

    int sm_num=0;  

    if(mon_runs>0){

      // for run number runa we do something 

      int run=0;

      cout <<" retrieve the data for a given run"<< endl;
      // retrieve the data for a given run
      cout << "here is the run number: "<< mon_run_vec[run].getRunIOV().getRunNumber() << endl;
      cout << "intrinsic counter: " << run << endl;
      
      RunIOV runiov_prime = mon_run_vec[run].getRunIOV();
      

      map<EcalLogicID, MonPedestalsDat> dataset_ped;
      econn->fetchDataSet(&dataset_ped, &mon_run_vec[run]);
      cout << "just retrieved data " <<endl;


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

      cout << "before unpacking " <<endl;

      for (CImped p = dataset_ped.begin(); p != dataset_ped.end(); p++) {
	ecid_xt = p->first;
	rd_ped  = p->second;
	sm_num=ecid_xt.getID1();
	int xt_num=ecid_xt.getID2()+(sm_num-1)*1700-1;
	ped_m12[xt_num]=rd_ped.getPedMeanG12();
	ped_m6[xt_num]=rd_ped.getPedMeanG6();
	ped_m1[xt_num]=rd_ped.getPedMeanG1();
       if(xt_num==123&&sm_num==10) cout <<"here is one value for XT: "<< xt_num<<" " << ped_m12[xt_num] << endl;
      }
      //  here we can do something to the pedestals like checking
      //  that they are correct and that there are no missing channels


      // then we ship them to the config DB

      cout << "after unpacking " <<endl;


      int mon_iov_id=mon_run_vec[run].getID();


      cout << "now create fe record " <<endl;

      FEConfigPedInfo fe_ped_info ;
      fe_ped_info.setIOVId(mon_iov_id);
      fe_ped_info.setTag("from_CondDB");
      econn->insertFEConfigPedInfo(fe_ped_info);

      cout << "just created record1 " <<endl;
      cout << "just created record2 " <<endl;
      cout << "just created record3 " <<endl;
      cout << "just created record4 " <<endl;
      cout << "just created record5 " <<endl;


      // this is to insert in FE Config DB      
      vector<EcalLogicID> ecid_vec;
      ecid_vec = econn->getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);

      cout << "just retrieved logic id " <<endl;

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
      cout << "Inserting dataset..." << flush;
      econn->insertDataArraySet(&dataset, &fe_ped_info);
      cout << "Done." << endl;
    
    }
    
  }





  void testRead(int iconf_req ) {
    // now we do something else 
    // this is an example for reading the pedestals 
    // for a given config iconf_req 
 
 
    FEConfigPedInfo fe_ped_info = econn->fetchFEConfigPedInfo(iconf_req);
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
    CondDBApp app( sid, user, pass);

    app.test(min_run, n_run);
    int i=1;
    app.testRead(i);
  } catch (exception &e) {
    cout << "ERROR:  " << e.what() << endl;
  } catch (...) {
    cout << "Unknown error caught" << endl;
  }

  cout << "All Done." << endl;

  return 0;
}
