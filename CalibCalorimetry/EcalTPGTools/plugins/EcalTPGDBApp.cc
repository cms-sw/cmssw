#include "CalibCalorimetry/EcalTPGTools/plugins/EcalTPGDBApp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <ctime>

using namespace std;
using namespace oracle::occi;

EcalTPGDBApp::EcalTPGDBApp(string host, string sid, string user, string pass, int port)
    : EcalCondDBInterface(host, sid, user, pass, port) {}

EcalTPGDBApp::EcalTPGDBApp(string sid, string user, string pass) : EcalCondDBInterface(sid, user, pass) {}

int EcalTPGDBApp::writeToConfDB_TPGPedestals(const map<EcalLogicID, FEConfigPedDat>& pedset, int iovId, string tag) {
  int result = 0;

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "******** Inserting Peds in conf-OMDS*****";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";

  edm::LogVerbatim("ECALTPGDBApp") << "creating fe record ";
  FEConfigPedInfo fe_ped_info;
  fe_ped_info.setIOVId(iovId);
  fe_ped_info.setConfigTag(tag);
  insertConfigSet(&fe_ped_info);
  result = fe_ped_info.getID();

  // Insert the dataset, identifying by iov
  edm::LogVerbatim("ECALTPGDBApp") << "*********about to insert peds *********";
  edm::LogVerbatim("ECALTPGDBApp") << " map size = " << pedset.size();
  insertDataArraySet(&pedset, &fe_ped_info);
  edm::LogVerbatim("ECALTPGDBApp") << "*********Done peds            *********";

  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGLinearCoef(const map<EcalLogicID, FEConfigLinDat>& linset,
                                              const map<EcalLogicID, FEConfigLinParamDat>& linparamset,
                                              int iovId,
                                              string tag) {
  int result = 0;

  edm::LogVerbatim("ECALTPGDBApp") << "*********************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "**Inserting Linarization coeff in conf-OMDS**";
  edm::LogVerbatim("ECALTPGDBApp") << "*********************************************";

  edm::LogVerbatim("ECALTPGDBApp") << "creating fe record ";
  FEConfigLinInfo fe_lin_info;
  fe_lin_info.setIOVId(iovId);
  fe_lin_info.setConfigTag(tag);
  insertConfigSet(&fe_lin_info);
  result = fe_lin_info.getID();

  // Insert the dataset, identifying by iov
  edm::LogVerbatim("ECALTPGDBApp") << "*********about to insert linearization coeff *********";
  edm::LogVerbatim("ECALTPGDBApp") << " map size = " << linset.size();
  insertDataArraySet(&linset, &fe_lin_info);
  insertDataArraySet(&linparamset, &fe_lin_info);
  edm::LogVerbatim("ECALTPGDBApp") << "*********Done lineraization coeff            *********";

  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGMain(int ped,
                                        int lin,
                                        int lut,
                                        int fgr,
                                        int sli,
                                        int wei,
                                        int wei2,
                                        int spi,
                                        int tim,
                                        int bxt,
                                        int btt,
                                        int bst,
                                        int cok,
                                        string tag,
                                        int ver) {
  int result = 0;

  edm::LogVerbatim("ECALTPGDBApp") << "*********************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "**Inserting Main FE table in conf-OMDS     **";
  edm::LogVerbatim("ECALTPGDBApp") << "*********************************************";

  edm::LogVerbatim("ECALTPGDBApp") << "creating fe record ";

  FEConfigMainInfo fe_main;
  fe_main.setPedId(ped);
  fe_main.setLinId(lin);
  fe_main.setLUTId(lut);
  fe_main.setFgrId(fgr);
  fe_main.setSliId(sli);
  fe_main.setWeiId(wei);
  fe_main.setWei2Id(wei2);
  fe_main.setSpiId(spi);
  fe_main.setTimId(tim);
  fe_main.setBxtId(bxt);
  fe_main.setBttId(btt);
  fe_main.setBstId(bst);
  fe_main.setCokeId(cok);
  fe_main.setConfigTag(tag);
  fe_main.setVersion(ver);

  insertConfigSet(&fe_main);
  result = fe_main.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*********Done Main           *********";

  return result;
}

void EcalTPGDBApp::readFromConfDB_TPGPedestals(int iconf_req) {
  // now we do something else
  // this is an example for reading the pedestals
  // for a given config iconf_req

  // FC alternatively a config set can be retrieved by the tag and version

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "test readinf fe_ped with id=" << iconf_req;
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";

  FEConfigPedInfo fe_ped_info;
  fe_ped_info.setId(iconf_req);

  fetchConfigSet(&fe_ped_info);

  map<EcalLogicID, FEConfigPedDat> dataset_ped;
  fetchDataSet(&dataset_ped, &fe_ped_info);

  typedef map<EcalLogicID, FEConfigPedDat>::const_iterator CIfeped;
  EcalLogicID ecid_xt;
  FEConfigPedDat rd_ped;

  for (CIfeped p = dataset_ped.begin(); p != dataset_ped.end(); ++p) {
    ecid_xt = p->first;
    rd_ped = p->second;
    //int sm_num=ecid_xt.getID1();
    ecid_xt.getID2();
    rd_ped.getPedMeanG12();
    rd_ped.getPedMeanG6();
    rd_ped.getPedMeanG1();
  }

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "test read done" << iconf_req;
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
}

int EcalTPGDBApp::readFromCondDB_Pedestals(map<EcalLogicID, MonPedestalsDat>& dataset, int runNb) {
  int iovId = 0;

  edm::LogVerbatim("ECALTPGDBApp") << "Retrieving run list from DB from run nb ... " << runNb;
  RunTag my_runtag;
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

  edm::LogVerbatim("ECALTPGDBApp") << "we are in read ped from condDB and runNb is " << runNb;

  mon_list = fetchMonRunListLastNRuns(my_runtag, montag, runNb, 10);

  edm::LogVerbatim("ECALTPGDBApp") << "we are in read ped from condDB";

  std::vector<MonRunIOV> mon_run_vec = mon_list.getRuns();
  edm::LogVerbatim("ECALTPGDBApp") << "number of ped runs is : " << mon_run_vec.size();
  int mon_runs = mon_run_vec.size();
  //int sm_num = 0;

  if (mon_runs > 0) {
    for (int ii = 0; ii < (int)mon_run_vec.size(); ii++)
      edm::LogVerbatim("ECALTPGDBApp") << "here is the run number: " << mon_run_vec[ii].getRunIOV().getRunNumber();

    // for the first run of the list we retrieve the pedestals
    int run = 0;
    edm::LogVerbatim("ECALTPGDBApp") << " retrieve the data for a given run";
    edm::LogVerbatim("ECALTPGDBApp") << "here is the run number: " << mon_run_vec[run].getRunIOV().getRunNumber();
    iovId = mon_run_vec[run].getID();

    fetchDataSet(&dataset, &mon_run_vec[run]);
  }
  return iovId;
}

int EcalTPGDBApp::writeToConfDB_TPGSliding(const map<EcalLogicID, FEConfigSlidingDat>& sliset, int iovId, string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting SLIDING************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  int result = 0;

  FEConfigSlidingInfo fe_info;
  fe_info.setIOVId(iovId);
  fe_info.setConfigTag(tag);
  insertConfigSet(&fe_info);

  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();

  // Insert the dataset
  insertDataArraySet(&sliset, &fe_info);

  result = fe_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************SLI done*********************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGLUT(const map<EcalLogicID, FEConfigLUTGroupDat>& lutgroupset,
                                       const map<EcalLogicID, FEConfigLUTDat>& lutset,
                                       const map<EcalLogicID, FEConfigLUTParamDat>& lutparamset,
                                       int iovId,
                                       string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting LUT************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  int result = 0;

  FEConfigLUTInfo fe_lut_info;
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

  result = fe_lut_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************LUT done*********************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGWeight(const map<EcalLogicID, FEConfigWeightGroupDat>& lutgroupset,
                                          const map<EcalLogicID, FEConfigWeightDat>& lutset,
                                          int ngr,
                                          string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting weights************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";

  int result = 0;

  FEConfigWeightInfo fe_wei_info;
  fe_wei_info.setNumberOfGroups(ngr);
  fe_wei_info.setConfigTag(tag);
  insertConfigSet(&fe_wei_info);

  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();

  // Insert the dataset
  insertDataArraySet(&lutgroupset, &fe_wei_info);
  // Insert the dataset
  insertDataArraySet(&lutset, &fe_wei_info);

  result = fe_wei_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************WEIGHT done******************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGWeight_doubleWeights(const map<EcalLogicID, FEConfigOddWeightGroupDat>& lutgroupset,
                                                        const map<EcalLogicID, FEConfigOddWeightDat>& lutset,
                                                        const map<EcalLogicID, FEConfigOddWeightModeDat>& tpmode,
                                                        int ngr,
                                                        string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting odd weights************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";

  int result = 0;

  FEConfigOddWeightInfo fe_wei_info;
  fe_wei_info.setNumberOfGroups(ngr);
  fe_wei_info.setConfigTag(tag);
  insertConfigSet(&fe_wei_info);

  //  Tm tdb = fe_lut_info.getDBTime();
  //tdb.dumpTm();

  // Insert the dataset
  insertDataArraySet(&lutgroupset, &fe_wei_info);
  // Insert the dataset
  insertDataArraySet(&lutset, &fe_wei_info);
  // Insert the Tpmode
  insertDataSet(&tpmode, &fe_wei_info);

  result = fe_wei_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************ODD WEIGHT + TPmode done******************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_TPGFgr(const map<EcalLogicID, FEConfigFgrGroupDat>& fgrgroupset,
                                       const map<EcalLogicID, FEConfigFgrDat>& fgrset,
                                       const map<EcalLogicID, FEConfigFgrParamDat>& fgrparamset,
                                       const map<EcalLogicID, FEConfigFgrEETowerDat>& dataset3,
                                       const map<EcalLogicID, FEConfigFgrEEStripDat>& dataset4,
                                       int iovId,
                                       string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting Fgr************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  int result = 0;

  FEConfigFgrInfo fe_fgr_info;
  fe_fgr_info.setNumberOfGroups(iovId);  // this eventually refers to some other table
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

  result = fe_fgr_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Fgr done*********************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_Spike(const map<EcalLogicID, FEConfigSpikeDat>& spikegroupset, string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting Spike************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  int result = 0;

  FEConfigSpikeInfo fe_spike_info;
  fe_spike_info.setConfigTag(tag);
  insertConfigSet(&fe_spike_info);

  //  Tm tdb = fe_fgr_info.getDBTime();
  //tdb.dumpTm();

  // Insert the dataset
  insertDataArraySet(&spikegroupset, &fe_spike_info);

  result = fe_spike_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Spike done*******************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

int EcalTPGDBApp::writeToConfDB_Delay(const map<EcalLogicID, FEConfigTimingDat>& timegroupset, string tag) {
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Inserting Delays************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  int result = 0;

  FEConfigTimingInfo fe_time_info;
  fe_time_info.setConfigTag(tag);
  insertConfigSet(&fe_time_info);

  //  Tm tdb = fe_fgr_info.getDBTime();
  //tdb.dumpTm();

  // Insert the dataset
  insertDataArraySet(&timegroupset, &fe_time_info);

  result = fe_time_info.getId();

  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  edm::LogVerbatim("ECALTPGDBApp") << "************Delays done******************";
  edm::LogVerbatim("ECALTPGDBApp") << "*****************************************";
  return result;
}

void EcalTPGDBApp::printTag(const RunTag* tag) const {
  edm::LogVerbatim("ECALTPGDBApp") << "=============RunTag:";
  edm::LogVerbatim("ECALTPGDBApp") << "GeneralTag:         " << tag->getGeneralTag();
  edm::LogVerbatim("ECALTPGDBApp") << "Location:           " << tag->getLocationDef().getLocation();
  edm::LogVerbatim("ECALTPGDBApp") << "Run Type:           " << tag->getRunTypeDef().getRunType();
  edm::LogVerbatim("ECALTPGDBApp") << "====================";
}

void EcalTPGDBApp::printIOV(const RunIOV* iov) const {
  edm::LogVerbatim("ECALTPGDBApp") << "=============RunIOV:";
  RunTag tag = iov->getRunTag();
  printTag(&tag);
  edm::LogVerbatim("ECALTPGDBApp") << "Run Number:         " << iov->getRunNumber();
  edm::LogVerbatim("ECALTPGDBApp") << "Run Start:          " << iov->getRunStart().str();
  edm::LogVerbatim("ECALTPGDBApp") << "Run End:            " << iov->getRunEnd().str();
  edm::LogVerbatim("ECALTPGDBApp") << "====================";
}
