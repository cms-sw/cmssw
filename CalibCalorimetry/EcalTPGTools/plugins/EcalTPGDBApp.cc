#include "CalibCalorimetry/EcalTPGTools/plugins/EcalTPGDBApp.h"

#include <vector>
#include <time.h>

using namespace std;
using namespace oracle::occi;

EcalTPGDBApp::EcalTPGDBApp(std::string host, std::string sid, std::string user, std::string pass, int port)
  : EcalCondDBInterface( host, sid, user, pass, port )
{}
 
EcalTPGDBApp::EcalTPGDBApp(std::string sid, std::string user, std::string pass)
  : EcalCondDBInterface(  sid, user, pass )
{}

int EcalTPGDBApp::writeToConfDB_TPGPedestals(const  std::map<EcalLogicID, FEConfigPedDat> & pedset, int iovId, std::string tag) {
  
  int result=0;

  std::cout << "*****************************************" << std::endl;
  std::cout << "******** Inserting Peds in conf-OMDS*****" << std::endl;
  std::cout << "*****************************************" << std::endl;
  
  std::cout << "creating fe record " <<std::endl;
  FEConfigPedInfo fe_ped_info ;
  fe_ped_info.setIOVId(iovId) ;
  fe_ped_info.setConfigTag(tag) ;
  insertConfigSet(&fe_ped_info) ;
  result = fe_ped_info.getID() ;


  // Insert the dataset, identifying by iov
  std::cout << "*********about to insert peds *********" << std::endl;
  std::cout << " map size = "<<pedset.size()<<std::endl ;
  insertDataArraySet(&pedset, &fe_ped_info);
  std::cout << "*********Done peds            *********" << std::endl;
  
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGLinearCoef(const  std::map<EcalLogicID, FEConfigLinDat> & linset, 
					      const  std::map<EcalLogicID, FEConfigLinParamDat> & linparamset, int iovId, std::string tag) {
  
  int result=0;

  std::cout << "*********************************************" << std::endl;
  std::cout << "**Inserting Linarization coeff in conf-OMDS**" << std::endl;
  std::cout << "*********************************************" << std::endl;
  
  std::cout << "creating fe record " <<std::endl;
  FEConfigLinInfo fe_lin_info ;
  fe_lin_info.setIOVId(iovId) ;
  fe_lin_info.setConfigTag(tag) ;
  insertConfigSet(&fe_lin_info) ;
  result = fe_lin_info.getID() ;
  
  // Insert the dataset, identifying by iov
  std::cout << "*********about to insert linearization coeff *********" << std::endl;
  std::cout << " map size = "<<linset.size()<<std::endl ;
  insertDataArraySet(&linset, &fe_lin_info);
  insertDataArraySet(&linparamset, &fe_lin_info);
  std::cout << "*********Done lineraization coeff            *********" << std::endl;
  
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGMain(int ped, int lin, int lut, int fgr, int sli, int wei, int bxt, int btt, std::string tag, int ver) {
  
  int result=0;

  std::cout << "*********************************************" << std::endl;
  std::cout << "**Inserting Main FE table in conf-OMDS     **" << std::endl;
  std::cout << "*********************************************" << std::endl;
  
  std::cout << "creating fe record " <<std::endl;

  FEConfigMainInfo fe_main ;
  fe_main.setPedId(ped) ;
  fe_main.setLinId(lin) ;
  fe_main.setLUTId(lut) ;
  fe_main.setFgrId(fgr) ;
  fe_main.setSliId(sli) ;
  fe_main.setWeiId(wei) ;
  fe_main.setBxtId(bxt) ;
  fe_main.setBttId(btt) ;
  fe_main.setConfigTag(tag) ;
  fe_main.setVersion(ver) ;

  insertConfigSet(&fe_main) ;
  result = fe_main.getId() ;
  
  std::cout << "*********Done Main           *********" << std::endl;
  
  return result;
}


void EcalTPGDBApp::readFromConfDB_TPGPedestals(int iconf_req ) {
  // now we do something else 
  // this is an example for reading the pedestals 
  // for a given config iconf_req 

  // FC alternatively a config set can be retrieved by the tag and version
  
  std::cout << "*****************************************" << std::endl;
  std::cout << "test readinf fe_ped with id="<<iconf_req  << std::endl;
  std::cout << "*****************************************" << std::endl;
  
  FEConfigPedInfo fe_ped_info;
  fe_ped_info.setId(iconf_req);

  fetchConfigSet(&fe_ped_info);

  std::map<EcalLogicID, FEConfigPedDat> dataset_ped;
  fetchDataSet(&dataset_ped, &fe_ped_info);
  
  typedef std::map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
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
    //int sm_num=ecid_xt.getID1();
    int xt_num=ecid_xt.getID2();
    ped_m12[xt_num]=rd_ped.getPedMeanG12();
    ped_m6[xt_num]=rd_ped.getPedMeanG6();
    ped_m1[xt_num]=rd_ped.getPedMeanG1();
  }
  
  std::cout << "*****************************************" << std::endl;
  std::cout << "test read done"<<iconf_req  << std::endl;
  std::cout << "*****************************************" << std::endl;
  
}


int EcalTPGDBApp::readFromCondDB_Pedestals(std::map<EcalLogicID, MonPedestalsDat> & dataset, int runNb) {


  int iovId = 0 ;
  
  std::cout << "Retrieving run list from DB from run nb ... "<< runNb << std::endl;
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

  std::cout<<"we are in read ped from condDB and runNb is "<< runNb<<std::endl;

  mon_list = fetchMonRunListLastNRuns(my_runtag, montag, runNb , 10 );

  std::cout<<"we are in read ped from condDB"<<std::endl;

  std::vector<MonRunIOV> mon_run_vec =  mon_list.getRuns();
  std::cout <<"number of ped runs is : "<< mon_run_vec.size()<< std::endl;
  int mon_runs = mon_run_vec.size();  
  //int sm_num = 0;  

  if(mon_runs>0) {
    for (int ii=0 ; ii<(int)mon_run_vec.size(); ii++) std::cout << "here is the run number: "<< mon_run_vec[ii].getRunIOV().getRunNumber() << std::endl;
    
    // for the first run of the list we retrieve the pedestals
    int run=0;
    std::cout <<" retrieve the data for a given run"<< std::endl;
    std::cout << "here is the run number: "<< mon_run_vec[run].getRunIOV().getRunNumber() << std::endl;
    iovId = mon_run_vec[run].getID();
    
    fetchDataSet(&dataset, &mon_run_vec[run]) ;   
  }
  return iovId ;
}


int EcalTPGDBApp::writeToConfDB_TPGSliding(const  std::map<EcalLogicID, FEConfigSlidingDat> & sliset, int iovId, std::string tag) 
{
  std::cout << "*****************************************" << std::endl;
  std::cout << "************Inserting SLIDING************" << std::endl;
  std::cout << "*****************************************" << std::endl;
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

  std::cout << "*****************************************" << std::endl;
  std::cout << "************SLI done*********************" << std::endl;
  std::cout << "*****************************************" << std::endl;
  return result;

}

int EcalTPGDBApp::writeToConfDB_TPGLUT(const  std::map<EcalLogicID, FEConfigLUTGroupDat> & lutgroupset,
					const  std::map<EcalLogicID, FEConfigLUTDat> & lutset, 
				       const  std::map<EcalLogicID, FEConfigLUTParamDat> & lutparamset,int iovId, std::string tag) 
{
  std::cout << "*****************************************" << std::endl;
  std::cout << "************Inserting LUT************" << std::endl;
  std::cout << "*****************************************" << std::endl;
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

  std::cout << "*****************************************" << std::endl;
  std::cout << "************LUT done*********************" << std::endl;
  std::cout << "*****************************************" << std::endl;
  return result;

}

int EcalTPGDBApp::writeToConfDB_TPGWeight(const  std::map<EcalLogicID, FEConfigWeightGroupDat> & lutgroupset,
					const  std::map<EcalLogicID, FEConfigWeightDat> & lutset, int ngr, std::string tag) 
{  
  std::cout << "*****************************************" << std::endl;
  std::cout << "************Inserting weights************" << std::endl;
  std::cout << "*****************************************" << std::endl;
  
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

  std::cout << "*****************************************" << std::endl;
  std::cout << "************WEIGHT done******************" << std::endl;
  std::cout << "*****************************************" << std::endl;
  return result;

  
}


int EcalTPGDBApp::writeToConfDB_TPGFgr(const  std::map<EcalLogicID, FEConfigFgrGroupDat> & fgrgroupset,
				       const  std::map<EcalLogicID, FEConfigFgrDat> & fgrset,  
				       const  std::map<EcalLogicID, FEConfigFgrParamDat> & fgrparamset,
				       const  std::map<EcalLogicID, FEConfigFgrEETowerDat> & dataset3, 
				       const  std::map<EcalLogicID, FEConfigFgrEEStripDat> & dataset4,
				       int iovId, std::string tag) 
{
  std::cout << "*****************************************" << std::endl;
  std::cout << "************Inserting Fgr************" << std::endl;
  std::cout << "*****************************************" << std::endl;
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

  std::cout << "*****************************************" << std::endl;
  std::cout << "************Fgr done*********************" << std::endl;
  std::cout << "*****************************************" << std::endl;
  return result;

}



void EcalTPGDBApp::printTag( const RunTag* tag) const
{
  std::cout << std::endl;
  std::cout << "=============RunTag:" << std::endl;
  std::cout << "GeneralTag:         " << tag->getGeneralTag() << std::endl;
  std::cout << "Location:           " << tag->getLocationDef().getLocation() << std::endl;
  std::cout << "Run Type:           " << tag->getRunTypeDef().getRunType() << std::endl;
  std::cout << "====================" << std::endl;
}

void EcalTPGDBApp::printIOV( const RunIOV* iov) const
{
  std::cout << std::endl;
  std::cout << "=============RunIOV:" << std::endl;
  RunTag tag = iov->getRunTag();
  printTag(&tag);
  std::cout << "Run Number:         " << iov->getRunNumber() << std::endl;
  std::cout << "Run Start:          " << iov->getRunStart().str() << std::endl;
  std::cout << "Run End:            " << iov->getRunEnd().str() << std::endl;
  std::cout << "====================" << std::endl;
}


