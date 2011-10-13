#include "CalibCalorimetry/EcalTPGTools/plugins/EcalTPGDBApp.h"

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
  fe_ped_info.setConfigTag(tag) ;
  insertConfigSet(&fe_ped_info) ;
  result = fe_ped_info.getID() ;


  // Insert the dataset, identifying by iov
  cout << "*********about to insert peds *********" << endl;
  cout << " map size = "<<pedset.size()<<endl ;
  insertDataArraySet(&pedset, &fe_ped_info);
  cout << "*********Done peds            *********" << endl;
  
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGLinearCoef(const  map<EcalLogicID, FEConfigLinDat> & linset, 
					      const  map<EcalLogicID, FEConfigLinParamDat> & linparamset, int iovId, string tag) {
  
  int result=0;

  cout << "*********************************************" << endl;
  cout << "**Inserting Linarization coeff in conf-OMDS**" << endl;
  cout << "*********************************************" << endl;
  
  cout << "creating fe record " <<endl;
  FEConfigLinInfo fe_lin_info ;
  fe_lin_info.setIOVId(iovId) ;
  fe_lin_info.setConfigTag(tag) ;
  insertConfigSet(&fe_lin_info) ;
  result = fe_lin_info.getID() ;
  
  // Insert the dataset, identifying by iov
  cout << "*********about to insert linearization coeff *********" << endl;
  cout << " map size = "<<linset.size()<<endl ;
  insertDataArraySet(&linset, &fe_lin_info);
  insertDataArraySet(&linparamset, &fe_lin_info);
  cout << "*********Done lineraization coeff            *********" << endl;
  
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGMain(int ped, int lin, int lut, int fgr, int sli, int wei, int spi, int tim, int bxt, int btt, int bst, string tag, int ver) {
  
  int result=0;

  cout << "*********************************************" << endl;
  cout << "**Inserting Main FE table in conf-OMDS     **" << endl;
  cout << "*********************************************" << endl;
  
  cout << "creating fe record " <<endl;

  FEConfigMainInfo fe_main ;
  fe_main.setPedId(ped) ;
  fe_main.setLinId(lin) ;
  fe_main.setLUTId(lut) ;
  fe_main.setFgrId(fgr) ;
  fe_main.setSliId(sli) ;
  fe_main.setWeiId(wei) ;
  fe_main.setSpiId(spi) ;
  fe_main.setTimId(tim) ;
  fe_main.setBxtId(bxt) ;
  fe_main.setBttId(btt) ;
  fe_main.setBstId(bst) ;
  fe_main.setConfigTag(tag) ;
  fe_main.setVersion(ver) ;

  insertConfigSet(&fe_main) ;
  result = fe_main.getId() ;
  
  cout << "*********Done Main           *********" << endl;
  
  return result;
}


void EcalTPGDBApp::readFromConfDB_TPGPedestals(int iconf_req ) {
  // now we do something else 
  // this is an example for reading the pedestals 
  // for a given config iconf_req 

  // FC alternatively a config set can be retrieved by the tag and version
  
  cout << "*****************************************" << endl;
  cout << "test readinf fe_ped with id="<<iconf_req  << endl;
  cout << "*****************************************" << endl;
  
  FEConfigPedInfo fe_ped_info;
  fe_ped_info.setId(iconf_req);

  fetchConfigSet(&fe_ped_info);

  map<EcalLogicID, FEConfigPedDat> dataset_ped;
  fetchDataSet(&dataset_ped, &fe_ped_info);
  
  typedef map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
  EcalLogicID ecid_xt;
  FEConfigPedDat  rd_ped;
  
  for (CIfeped p = dataset_ped.begin(); p != dataset_ped.end(); p++) {
    ecid_xt = p->first;
    rd_ped  = p->second;
    //int sm_num=ecid_xt.getID1();
    ecid_xt.getID2();
    rd_ped.getPedMeanG12();
    rd_ped.getPedMeanG6();
    rd_ped.getPedMeanG1();
  }
  
  cout << "*****************************************" << endl;
  cout << "test read done"<<iconf_req  << endl;
  cout << "*****************************************" << endl;
  
}


int EcalTPGDBApp::readFromCondDB_Pedestals(map<EcalLogicID, MonPedestalsDat> & dataset, int runNb) {


  int iovId = 0 ;
  
  cout << "Retrieving run list from DB from run nb ... "<< runNb << endl;
  RunTag  my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;
  my_locdef.setLocation("P5_Co");
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

  std::cout<<"we are in read ped from condDB and runNb is "<< runNb<<endl;

  mon_list = fetchMonRunListLastNRuns(my_runtag, montag, runNb , 10 );

  std::cout<<"we are in read ped from condDB"<<endl;

  std::vector<MonRunIOV> mon_run_vec =  mon_list.getRuns();
  cout <<"number of ped runs is : "<< mon_run_vec.size()<< endl;
  int mon_runs = mon_run_vec.size();  
  //int sm_num = 0;  

  if(mon_runs>0) {
    for (int ii=0 ; ii<(int)mon_run_vec.size(); ii++) cout << "here is the run number: "<< mon_run_vec[ii].getRunIOV().getRunNumber() << endl;
    
    // for the first run of the list we retrieve the pedestals
    int run=0;
    cout <<" retrieve the data for a given run"<< endl;
    cout << "here is the run number: "<< mon_run_vec[run].getRunIOV().getRunNumber() << endl;
    iovId = mon_run_vec[run].getID();
    
    fetchDataSet(&dataset, &mon_run_vec[run]) ;   
  }
  return iovId ;
}


int EcalTPGDBApp::writeToConfDB_TPGSliding(const  map<EcalLogicID, FEConfigSlidingDat> & sliset, int iovId, string tag) 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting SLIDING************" << endl;
  cout << "*****************************************" << endl;
  int result=0; 

  FEConfigSlidingInfo fe_info ;
  fe_info.setIOVId(iovId); 
  fe_info.setConfigTag(tag);
  insertConfigSet(&fe_info);
  
  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&sliset, &fe_info);

  result=fe_info.getId();

  cout << "*****************************************" << endl;
  cout << "************SLI done*********************" << endl;
  cout << "*****************************************" << endl;
  return result;

}


int EcalTPGDBApp::writeToConfDB_TPGLUT(const  map<EcalLogicID, FEConfigLUTGroupDat> & lutgroupset,
					const  map<EcalLogicID, FEConfigLUTDat> & lutset, 
				       const  map<EcalLogicID, FEConfigLUTParamDat> & lutparamset,int iovId, string tag) 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting LUT************" << endl;
  cout << "*****************************************" << endl;
  int result=0; 

  FEConfigLUTInfo fe_lut_info ;
  fe_lut_info.setNumberOfGroups(iovId); 
  fe_lut_info.setConfigTag(tag);
  insertConfigSet(&fe_lut_info);
  
  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&lutgroupset, &fe_lut_info);
  // Insert the dataset
  insertDataArraySet(&lutset, &fe_lut_info);
  // insert the parameters
  insertDataArraySet(&lutparamset, &fe_lut_info);
  
  result=fe_lut_info.getId();

  cout << "*****************************************" << endl;
  cout << "************LUT done*********************" << endl;
  cout << "*****************************************" << endl;
  return result;

}

int EcalTPGDBApp::writeToConfDB_TPGWeight(const  map<EcalLogicID, FEConfigWeightGroupDat> & lutgroupset,
					const  map<EcalLogicID, FEConfigWeightDat> & lutset, int ngr, string tag) 
{  
  cout << "*****************************************" << endl;
  cout << "************Inserting weights************" << endl;
  cout << "*****************************************" << endl;
  
  int result=0; 

  FEConfigWeightInfo fe_wei_info ;
  fe_wei_info.setNumberOfGroups(5); // this eventually refers to some other table 
  fe_wei_info.setConfigTag(tag);
  insertConfigSet(&fe_wei_info);
  
  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&lutgroupset, &fe_wei_info);
  // Insert the dataset
  insertDataArraySet(&lutset, &fe_wei_info);
  
  result=fe_wei_info.getId();

  cout << "*****************************************" << endl;
  cout << "************WEIGHT done******************" << endl;
  cout << "*****************************************" << endl;
  return result;

  
}


int EcalTPGDBApp::writeToConfDB_TPGFgr(const  map<EcalLogicID, FEConfigFgrGroupDat> & fgrgroupset,
				       const  map<EcalLogicID, FEConfigFgrDat> & fgrset,  
				       const  map<EcalLogicID, FEConfigFgrParamDat> & fgrparamset,
				       const  map<EcalLogicID, FEConfigFgrEETowerDat> & dataset3, 
				       const  map<EcalLogicID, FEConfigFgrEEStripDat> & dataset4,
				       int iovId, string tag) 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting Fgr************" << endl;
  cout << "*****************************************" << endl;
  int result=0; 

  FEConfigFgrInfo fe_fgr_info ;
  fe_fgr_info.setNumberOfGroups(iovId); // this eventually refers to some other table 
  fe_fgr_info.setConfigTag(tag);
  insertConfigSet(&fe_fgr_info);
  
  //  Tm tdb = fe_fgr_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&fgrgroupset, &fe_fgr_info);
  // Insert the dataset
  insertDataArraySet(&fgrset, &fe_fgr_info);
  // Insert the parameters
  insertDataArraySet(&fgrparamset, &fe_fgr_info);
  // Insert the parameters
  insertDataArraySet(&dataset3, &fe_fgr_info);
  // Insert the parameters
  insertDataArraySet(&dataset4, &fe_fgr_info);
  
  result=fe_fgr_info.getId();

  cout << "*****************************************" << endl;
  cout << "************Fgr done*********************" << endl;
  cout << "*****************************************" << endl;
  return result;

}

int EcalTPGDBApp::writeToConfDB_Spike(const  map<EcalLogicID, FEConfigSpikeDat> & spikegroupset,
				       string tag) 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting Spike************" << endl;
  cout << "*****************************************" << endl;
  int result=0; 

  FEConfigSpikeInfo fe_spike_info ;
  fe_spike_info.setConfigTag(tag);
  insertConfigSet(&fe_spike_info);
  
  //  Tm tdb = fe_fgr_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&spikegroupset, &fe_spike_info);
  
  result=fe_spike_info.getId();

  cout << "*****************************************" << endl;
  cout << "************Spike done*******************" << endl;
  cout << "*****************************************" << endl;
  return result;

}

int EcalTPGDBApp::writeToConfDB_Delay(const  map<EcalLogicID, FEConfigTimingDat> & timegroupset,
				       string tag) 
{
  cout << "*****************************************" << endl;
  cout << "************Inserting Delays************" << endl;
  cout << "*****************************************" << endl;
  int result=0; 

  FEConfigTimingInfo fe_time_info ;
  fe_time_info.setConfigTag(tag);
  insertConfigSet(&fe_time_info);
  
  //  Tm tdb = fe_fgr_info.getDBTime();
  //tdb.dumpTm();
  
  // Insert the dataset
  insertDataArraySet(&timegroupset, &fe_time_info);
  
  result=fe_time_info.getId();

  cout << "*****************************************" << endl;
  cout << "************Delays done******************" << endl;
  cout << "*****************************************" << endl;
  return result;

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


