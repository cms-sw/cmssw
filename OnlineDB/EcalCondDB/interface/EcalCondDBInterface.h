/***********************************************/
/* EcalCondDBInterface.h		       */
/* 					       */
/* $Id: EcalCondDBInterface.h,v 1.4 2006/07/05 15:47:37 egeland Exp $ 	        		       */
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
#include <occi.h>

#include "OnlineDB/EcalCondDB/interface/EcalDBConnection.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/DCUIOV.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"

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
			      std::string mapsTo = ""
			      )
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
					      std::string mapsTo = ""
					      )
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
   *  Return a moniotring run object
   */
  MonRunIOV fetchMonRunIOV(RunTag* runtag, MonRunTag* montag, run_t run, subrun_t monrun)
    throw(std::runtime_error);


  /**
   *   Return a DCU IOV object
   */
  DCUIOV fetchDCUIOV(DCUTag* tag, Tm evenTm)
    throw(std::runtime_error);



  /**
   *  Return a laser moniotring farm run object
   */
  LMFRunIOV fetchLMFRunIOV(RunTag* runtag, LMFRunTag* lmftag, run_t run, subrun_t lmfrun)
    throw(std::runtime_error);



  /**
   *   Return a Calibration IOV object
   */
  CaliIOV fetchCaliIOV(CaliTag* tag, Tm evenTm)
    throw(std::runtime_error);



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

    datiface.fetchData( fillMap, iov );
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
    datiface.fetchData( fillMap, fillIOV );
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
