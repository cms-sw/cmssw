#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGDBApp.h"

#include <vector>
#include <time.h>

using namespace std;
using namespace oracle::occi;

EcalTPGDBApp::EcalTPGDBApp(string host, string sid, string user, string pass, int port)
  : EcalCondDBInterface( host, sid, user, pass, port )
{}
 
EcalTPGDBApp::EcalTPGDBApp(string sid, string user, string pass)
  : EcalCondDBInterface(  sid, user, pass )
{}

int EcalTPGDBApp::writeToConfDB_TPGPedestals(const  map<EcalLogicID, FEConfigPedDat> & pedset, int iovId, string tag) {
  
  int result=0;

  cout << "*****************************************" << endl;
  cout << "******** Inserting Peds in conf-OMDS*****" << endl;
  cout << "*****************************************" << endl;
  
  cout << "creating fe record " <<endl;
  FEConfigPedInfo fe_ped_info ;
  fe_ped_info.setIOVId(iovId) ;
  fe_ped_info.setTag(tag) ;
  insertFEConfigPedInfo(&fe_ped_info) ;
  result = fe_ped_info.getID() ;
  
  // Insert the dataset, identifying by iov
  cout << "*********about to insert peds *********" << endl;
  cout << " map size = "<<pedset.size()<<endl ;
  insertDataArraySet(&pedset, &fe_ped_info);
  cout << "*********Done peds            *********" << endl;
  
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGLinearCoef(const  map<EcalLogicID, FEConfigLinDat> & linset, int iovId, string tag) {
  
  int result=0;

  cout << "*********************************************" << endl;
  cout << "**Inserting Linarization coeff in conf-OMDS**" << endl;
  cout << "*********************************************" << endl;
  
  cout << "creating fe record " <<endl;
  FEConfigLinInfo fe_lin_info ;
  fe_lin_info.setIOVId(iovId) ;
  fe_lin_info.setTag(tag) ;
  //insertFEConfigLinInfo(&fe_lin_info) ;
  result = fe_lin_info.getID() ;
  
  // Insert the dataset, identifying by iov
  cout << "*********about to insert linearization coeff *********" << endl;
  cout << " map size = "<<linset.size()<<endl ;
  insertDataArraySet(&linset, &fe_lin_info);
  cout << "*********Done lineraization coeff            *********" << endl;
  
  return result;
}


void EcalTPGDBApp::readFromConfDB_TPGPedestals(int iconf_req ) {
  // now we do something else 
  // this is an example for reading the pedestals 
  // for a given config iconf_req 
  
  cout << "*****************************************" << endl;
  cout << "test readinf fe_ped with id="<<iconf_req  << endl;
  cout << "*****************************************" << endl;
  
  FEConfigPedInfo fe_ped_info = fetchFEConfigPedInfo(iconf_req);
  map<EcalLogicID, FEConfigPedDat> dataset_ped;
  fetchDataSet(&dataset_ped, &fe_ped_info);
  
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
  }
  
  cout << "*****************************************" << endl;
  cout << "test read done"<<iconf_req  << endl;
  cout << "*****************************************" << endl;
  
}


int EcalTPGDBApp::readFromCondDB_Pedestals(map<EcalLogicID, MonPedestalsDat> & dataset, int runNb) {

  int iovId = 0 ;
  
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
  mon_list = fetchMonRunListLastNRuns(my_runtag, montag, runNb , 10 );
  
  std::vector<MonRunIOV> mon_run_vec =  mon_list.getRuns();
  cout <<"number of ped runs is : "<< mon_run_vec.size()<< endl;
  int mon_runs = mon_run_vec.size();  
  int sm_num = 0;  

  if(mon_runs>0) {
    for (int ii=0 ; ii<mon_run_vec.size(); ii++) cout << "here is the run number: "<< mon_run_vec[ii].getRunIOV().getRunNumber() << endl;
    
    // for the first run of the list we retrieve the pedestals
    int run=0;
    cout <<" retrieve the data for a given run"<< endl;
    cout << "here is the run number: "<< mon_run_vec[run].getRunIOV().getRunNumber() << endl;
    iovId = mon_run_vec[run].getID();
    
    fetchDataSet(&dataset, &mon_run_vec[run]) ;   
  }
  return iovId ;
}


void EcalTPGDBApp::writeToConfDB_TPGLUT() 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting LUT************" << endl;
  cout << "*****************************************" << endl;
  
  FEConfigLUTInfo fe_lut_info ;
  fe_lut_info.setIOVId(0); // this eventually refers to some other table 
  fe_lut_info.setTag("test");
  insertFEConfigLUTInfo(&fe_lut_info);
  
  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();
  
  vector<EcalLogicID> ecid_vec;
  ecid_vec = getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);
  
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
  insertDataArraySet(&dataset, &fe_lut_info);
  
  
  
  // now we store in the DB the correspondence btw channels and LUT groups 
  map<EcalLogicID, FEConfigLUTDat> dataset2;
  // in this case I decide in a stupid way which channel belongs to which group 
  for (int ich=0; ich<ecid_vec.size() ; ich++){
    FEConfigLUTDat lut;
    int igroup=ich/(ecid_vec.size()/5);
    lut.setLUTGroupId(igroup);
    // Fill the dataset
    dataset2[ecid_vec[ich]] = lut;
  }
  
  // Insert the dataset
  
  insertDataArraySet(&dataset2, &fe_lut_info);
  cout << "*****************************************" << endl;
  cout << "************LUT done*********************" << endl;
  cout << "*****************************************" << endl;
  
}

void EcalTPGDBApp::writeToConfDB_TPGWeights(FEConfigWeightGroupDat & weight) 
{  
  cout << "*****************************************" << endl;
  cout << "************Inserting weights************" << endl;
  cout << "*****************************************" << endl;
  
  FEConfigWeightInfo fe_wei_info ;
  fe_wei_info.setNumberOfGroups(5); // this eventually refers to some other table 
  fe_wei_info.setTag("my preferred weights");
  insertFEConfigWeightInfo(&fe_wei_info);
  
  vector<EcalLogicID> ecid_vec;
  ecid_vec = getEcalLogicIDSet("EB_crystal_number", 1,36,1,1700);
  
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
  insertDataArraySet(&dataset, &fe_wei_info);
  
  
  
  // now we store in the DB the correspondence btw channels and LUT groups 
  map<EcalLogicID, FEConfigWeightDat> dataset2;
  // in this case I decide in a stupid way which channel belongs to which group 
  for (int ich=0; ich<ecid_vec.size() ; ich++){
    FEConfigWeightDat weid;
    int igroup=ich/(ecid_vec.size()/5);
    weid.setWeightGroupId(igroup);
    // Fill the dataset
    dataset2[ecid_vec[ich]] = weid;
  }
  
  // Insert the dataset
  
  insertDataArraySet(&dataset2, &fe_wei_info);
  cout << "*****************************************" << endl;
  cout << "************weights done*****************" << endl;
  cout << "*****************************************" << endl;
  
}



void EcalTPGDBApp::printTag( const RunTag* tag) const
{
  cout << endl;
  cout << "=============RunTag:" << endl;
  cout << "GeneralTag:         " << tag->getGeneralTag() << endl;
  cout << "Location:           " << tag->getLocationDef().getLocation() << endl;
  cout << "Run Type:           " << tag->getRunTypeDef().getRunType() << endl;
  cout << "====================" << endl;
}

void EcalTPGDBApp::printIOV( const RunIOV* iov) const
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


