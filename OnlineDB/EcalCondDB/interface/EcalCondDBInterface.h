/***********************************************/
/* EcalCondDBInterface.h		       */
/* 					       */
/* $Id: EcalCondDBInterface.h,v 1.10 2007/11/14 16:38:48 fra Exp $ 	        		       */
/* 					       */
/* Interface to the Ecal Conditions DB.	       */
/***********************************************/

#ifndef ECALCONDDBINTERFACE_HH
#define ECALCONDDBINTERFACE_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"


#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunList.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunList.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/DCSPTMTempList.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"


class EcalCondDBInterface : public EcalDBConnection {
 public:

  /******************\
  -  public methods  -
  \******************/

  /**
   *  Constructor, makes connection to DB without TNS_ADMIN
   *  === Parameters ===
   *  host:    DB host machine
   *  sid:     DB SID (name)
   *  user:    User to connect
   *  pass:    Password for user
   *  port:    port number to connect, default 1521
   */
  EcalCondDBInterface( std::string host,
		       std::string sid,
		       std::string user,
		       std::string pass,
		       int port=1521 )
    : EcalDBConnection( host, sid, user, pass, port )
    {
      // call the parent constructor

      // create a DateHandler
      dh = new DateHandler(env, conn);
    }



  /**
   *  Constructor, makes connection to DB with TNS_ADMIN
   *  === Parameters ===
   *  sid:     DB SID (name)
   *  user:    User to connect
   *  pass:    Password for user
   */
  EcalCondDBInterface( std::string sid,
		       std::string user,
		       std::string pass )
    : EcalDBConnection( sid, user, pass )
    {
      // call the parent constructor

      // create a DateHandler
      dh = new DateHandler(env, conn);
    }


  /**
   * Destructor
   */
  virtual ~EcalCondDBInterface()
    throw(std::runtime_error)
    {
      // call the parent destructor
      
      // destroy the DateHandler
      delete(dh);
    }



  /**
   *  Return a date handler associated with this connection
   */
  inline DateHandler* getDateHandler()
    {
      return dh;
    }

  
  /**
   *  Look up the "human readable" ids and return an EcalLogicID object which contains
   *  the "database readable" logic_id
   *  === Parameters ===
   *  name:           name of the channel type you are specifying
   *  id1, id2, id3:  ids of the channel type
   *  mapsTo:         name of the channel type you are mapping to
   */
  EcalLogicID getEcalLogicID( std::string name,
			      int id1 = EcalLogicID::NULLID,
			      int id2 = EcalLogicID::NULLID,
			      int id3 = EcalLogicID::NULLID,
			      std::string mapsTo = "" )
    throw(std::runtime_error);

  /**
   *  Look up the database logic_id and return the EcalLogicID object which contains
   *  the "human readable" ids
   *  === Parameters ===
   *  logicID:  DB logic_id
   */
  EcalLogicID getEcalLogicID( int logicID )
    throw(std::runtime_error);

  /**
   *  Get a set of EcalLogicID in one transaction
   *  === Parameters ===
   *  name:             name of the channel type you are specifying
   *  fromId1, toId1:   Range of id1 in the DB to retrieve
   *  fromId2, toId2:   Range of id2 in the DB to retrieve
   *  fromId3, toId3:   Range of id3 in the DB to retrieve
   *  string mapsTo:    channel type name these ids map to
   */  
  std::vector<EcalLogicID> getEcalLogicIDSet( std::string name,
					      int fromId1, int toId1,
					      int fromId2 = EcalLogicID::NULLID, int toId2 = EcalLogicID::NULLID,
					      int fromId3 = EcalLogicID::NULLID, int toId3 = EcalLogicID::NULLID,
					      std::string mapsTo = "" )
    throw(std::runtime_error);



  /**
   *  Insert a run IOV object.  Nothing is committed in the event of an exception
   */
  void insertRunIOV(RunIOV* iov)
    throw(std::runtime_error);


  /**
   *  Return a run IOV object for a given tag
   */
  RunIOV fetchRunIOV(RunTag* tag, run_t run)
    throw(std::runtime_error);



  /**
   *  Return a run IOV object for a given location.
   *  It is not guarunteed that a run is unique for a location only,
   *  so an exception is thrown if more than one result exists.
   */
  RunIOV fetchRunIOV(std::string location, run_t run)
    throw(std::runtime_error);



  /**
   *  Return a monitoring run object
   */
  MonRunIOV fetchMonRunIOV(RunTag* runtag, MonRunTag* montag, run_t run, subrun_t monrun)
    throw(std::runtime_error);


  /**
   *   Return a DCU IOV object
   */
  DCUIOV fetchDCUIOV(DCUTag* tag, Tm evenTm)
    throw(std::runtime_error);



  /**
   *  Return a laser monitoring farm run object
   */
  LMFRunIOV fetchLMFRunIOV(RunTag* runtag, LMFRunTag* lmftag, run_t run, subrun_t lmfrun)
    throw(std::runtime_error);



  /**
   *   Return a Calibration IOV object
   */
  CaliIOV fetchCaliIOV(CaliTag* tag, Tm evenTm)
    throw(std::runtime_error);


 /**
   *   Return a Run List
   */
  RunList fetchRunList(RunTag tag) throw(std::runtime_error);

 /**
   *   Return a PTM Temp List
   */

  DCSPTMTempList fetchDCSPTMTempList(EcalLogicID ecid)  throw(runtime_error);
  DCSPTMTempList fetchDCSPTMTempList(EcalLogicID ecid, Tm start, Tm end) throw(runtime_error);
 /**template<class DATT, class IOVT>
  void insertDataSet(const std::map< EcalLogicID, DATT >* data, IOVT* iov)
    throw(std::runtime_error)
  {
    try {
      iov->setConnection(env, conn);
      iov->writeDB();
      
      DATT dataIface;
      dataIface.setConnection(env, conn);
      dataIface.prepareWrite();
      
      const EcalLogicID* channel;
      const DATT* dataitem;
      typedef typename std::map< EcalLogicID, DATT >::const_iterator CI;
      for (CI p = data->begin(); p != data->end(); ++p) {
	channel = &(p->first);
	dataitem = &(p->second);
	dataIface.writeDB( channel, dataitem, iov);
      }
      conn->commit();
      dataIface.terminateWriteStatement();
    } catch (std::runtime_error &e) {
      conn->rollback();
      throw(e);
    } catch (...) {
      conn->rollback();
      throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
    }
  }
   *   Return a LMFRun List
   */
  LMFRunList fetchLMFRunList(RunTag tag, LMFRunTag lmfruntag) throw(std::runtime_error);
  LMFRunList fetchLMFRunList(RunTag tag, LMFRunTag lmfruntag,int min_run, int max_run) throw(std::runtime_error);
  LMFRunList fetchLMFRunListLastNRuns(RunTag tag, LMFRunTag lmfruntag, int max_run, int n_runs) throw(std::runtime_error);

  MonRunList fetchMonRunList(RunTag tag, MonRunTag monruntag) throw(std::runtime_error);
  MonRunList fetchMonRunList(RunTag tag, MonRunTag monruntag,int min_run, int max_run) throw(std::runtime_error);
  MonRunList fetchMonRunListLastNRuns(RunTag tag, MonRunTag monruntag, int max_run, int n_runs) throw(std::runtime_error);

  // methods for the config FE API
  FEConfigPedInfo fetchFEConfigPedInfo(int id)  throw(runtime_error);
  FEConfigPedInfo fetchFEConfigPedInfo(std::string tag ) throw(runtime_error);
  FEConfigPedInfo fetchFEConfigPedInfoLast()  throw(runtime_error);
  void insertFEConfigPedInfo(FEConfigPedInfo* iconf) throw(runtime_error);

  FEConfigLUTInfo fetchFEConfigLUTInfoByID(int id)  throw(runtime_error);
  FEConfigLUTInfo fetchFEConfigLUTInfo(int id)  throw(runtime_error);
  FEConfigLUTInfo fetchFEConfigLUTInfo(std::string tag ) throw(runtime_error);
  FEConfigLUTInfo fetchFEConfigLUTInfoLast()  throw(runtime_error);
  int insertFEConfigLUTInfo(FEConfigLUTInfo* iconf) throw(runtime_error);


  FEConfigWeightInfo fetchFEConfigWeightInfo(std::string tag ) throw(runtime_error);
  FEConfigWeightInfo fetchFEConfigWeightInfoLast()  throw(runtime_error);
  void insertFEConfigWeightInfo(FEConfigWeightInfo* iconf) throw(runtime_error);



  /*
   *   Insert a set of data at the given iov.  If the iov does not yet
   *   exist in the database they will be written.  Nothing is committed in the the event of
   *   an exception
   */
  // XXX TODO:  Split declaration and definition using a macro
  // XXX        Naive method causes linker errors...
  // XXX        See example FWCore/Framework/interface/eventSetupGetImplementation.h

  template<class DATT, class IOVT>
  void insertDataSet(const std::map< EcalLogicID, DATT >* data, IOVT* iov)
    throw(std::runtime_error)
  {
    try {
      iov->setConnection(env, conn);
      // if it has not yet been written then write 
      if(iov->getID()==0){
	cout<<"IOV was not set we retrieve it from DB"<<endl;
	iov->fetchID();
      } 
      if(iov->getID()==0){
	cout<<"IOV was not written we write it"<<endl;
	iov->writeDB();
      } 
      
      DATT dataIface;
      dataIface.setConnection(env, conn);
      dataIface.prepareWrite();
      
      const EcalLogicID* channel;
      const DATT* dataitem;
      typedef typename std::map< EcalLogicID, DATT >::const_iterator CI;
      for (CI p = data->begin(); p != data->end(); ++p) {
	channel = &(p->first);
	dataitem = &(p->second);
	dataIface.writeDB( channel, dataitem, iov);
      }
      conn->commit();
      dataIface.terminateWriteStatement();
    } catch (std::runtime_error &e) {
      conn->rollback();
      throw(e);
    } catch (...) {
      conn->rollback();
      throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
    }
  }
  
  //test for DB Array insertion
  template<class DATT, class IOVT>
  void insertDataArraySet(const std::map< EcalLogicID, DATT >* data, IOVT* iov)
    throw(std::runtime_error)
  {
    try {
      iov->setConnection(env, conn);
      if(iov->getID()==0){
	cout<<"IOV was not set we retrieve it from DB"<<endl;
	iov->fetchID();
      } 
      if(iov->getID()==0){
	cout<<"IOV was not written we write it"<<endl;
	iov->writeDB();
      } 

      
      DATT dataIface;
      dataIface.setConnection(env, conn);
      dataIface.prepareWrite();
      
      dataIface.writeArrayDB(data, iov);
      conn->commit();
      
      dataIface.terminateWriteStatement();
   
    } catch (std::runtime_error &e) {
      conn->rollback();
      throw(e);
    } catch (...) {
      conn->rollback();
      throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
    }
  }

 template<class DATT, class IOVT>
  void insertDataSetVector( std::vector<EcalLogicID> ecid, std::vector<IOVT> run_iov, std::vector<DATT> data )
    throw(std::runtime_error)
  {
   
    int nruns= run_iov.size();
   
    if(run_iov.size()!=ecid.size() &&ecid.size()!=data.size()){
      throw(std::runtime_error("EcalCondDBInterface::insertDataSetVector:  vector sizes are different.."));
    } 


    try {
      
      DATT dataIface;
      dataIface.setConnection(env, conn);
      dataIface.prepareWrite();

      for (int i=0; i<nruns; i++){

	run_iov[i].setConnection(env, conn);
	run_iov[i].writeDB();
      
	dataIface.writeDB( &ecid[i], &data[i], &run_iov[i]);
	
	conn->commit();
      }
    } catch (std::runtime_error &e) {
      conn->rollback();
      throw(e);
    } catch (...) {
      conn->rollback();
      throw(std::runtime_error("EcalCondDBInterface::insertDataSet:  Unknown exception caught"));
    }
  }




  /*
   *  Fetch a set of data based on an EXACT match of an iov
   */
  template<class DATT, class IOVT>
  void fetchDataSet(std::map< EcalLogicID, DATT >* fillMap, IOVT* iov)
    throw(std::runtime_error)
  {
    fillMap->clear();

    DATT datiface;
    datiface.setConnection(env, conn);
    datiface.createReadStatement();
    datiface.setPrefetchRowCount(1000);
    datiface.fetchData( fillMap, iov );
    datiface.terminateReadStatement();

  }

  /*
   *  Fetch dataset that is valid for the given RunTag and run.
   *  Also fills the given IOV object with the IOV associated with the data
   *  run is optional, by default is set to infinity
   *  Note:  ONLY works for Run*Dat objects
   *  TODO:  Make this function (or similar) work for all *Dat objects
   */
  template<class DATT, class IOVT>
  void fetchValidDataSet(std::map< EcalLogicID, DATT >* fillMap, 
			 IOVT* fillIOV, 
			 RunTag* tag, run_t run = (unsigned int)-1)
  throw(std::runtime_error)
  {
    fillMap->clear();
    DATT datiface;
    fillIOV->setConnection(env, conn);
    fillIOV->setByRecentData(datiface.getTable(), tag, run);
    datiface.setConnection(env, conn);
    datiface.createReadStatement();
    datiface.setPrefetchRowCount(1000);
    datiface.fetchData( fillMap, fillIOV );
    datiface.terminateReadStatement();
  }

  /*
   *  Fetch dataset that is valid for the given location and run.
   *  Also fills the given IOV object with the IOV associated with the data
   *  run is optional, by default is set to infinity
   *  Note:  ONLY works for Run*Dat objects
   *  TODO:  Make this function (or similar) work for all *Dat objects
   */
  template<class DATT, class IOVT>
  void fetchValidDataSet(std::map< EcalLogicID, DATT >* fillMap,
                         IOVT* fillIOV,
                         string location, run_t run = (unsigned int)-1)
  throw(std::runtime_error)
  {
    fillMap->clear();
    DATT datiface;
    fillIOV->setConnection(env, conn);
    fillIOV->setByRecentData(datiface.getTable(), location, run);
    datiface.setConnection(env, conn);
    datiface.createReadStatement();
    datiface.setPrefetchRowCount(1000);
    datiface.fetchData( fillMap, fillIOV );
    datiface.terminateReadStatement();
  }





  void dummy();

 private:

  /*********************\
  -  private variables  -
  \*********************/

  DateHandler* dh;

  EcalCondDBInterface();
  EcalCondDBInterface(const EcalCondDBInterface& copy);


};

#endif
