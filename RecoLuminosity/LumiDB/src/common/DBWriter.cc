/**
 *   DBWrite: Insertion, Selection, updating and deletion of 
 *   a row through .
 *
 *   Description
 *   Create a program which has insert, select, update & 
 *   delete as operations. 
 *   Perform all these operations using  interface.
 **/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>

#include "DBWriter.hh"

// Exception handler for DB
#include "OracleDBException.hh"

using namespace oracle::occi;
using namespace std;
using namespace ICCoreUtils;

namespace HCAL_HLX {

  DBWriter::DBWriter(const string & user,
		     const string & passwd,
		     const string & db,
		     const string & dbOwner) {
    _env = 0;
    _conn = 0;
    _dbOwner = "";
    try {
      // NOTE - must catch SQLException above this
      // in case DB is inaccessible
      cout << "------------------------------------" << endl;
      cout << "Creating OMDS connection" << endl;
      cout << "User: " << user << endl;
      cout << "Password: " << passwd << endl;
      cout << "DB: " << db << endl;
      cout << "DB owner: " << dbOwner << endl;
      cout << "------------------------------------" << endl;

      _env = Environment::createEnvironment (Environment::DEFAULT);
      _conn = _env->createConnection (user, passwd, db);
      _dbOwner = dbOwner;
    } catch (SQLException & aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  DBWriter::~DBWriter () {
    try {
      cout << "In " << __PRETTY_FUNCTION__ << endl;
      if ( _conn ) {
	cout << "Inccc " << __PRETTY_FUNCTION__ << endl;
	_env->terminateConnection (_conn);
	_conn = 0;
      }
      if ( _env ) {
	cout << "Ineee " << __PRETTY_FUNCTION__ << endl;
	Environment::terminateEnvironment (_env);
	_env = 0;
      }
    } catch (SQLException & aExc) {
      cout << "Exception raised" << endl;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  void DBWriter::save() {
    string sqlStmt = "commit";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);      
    }
  }

  
  /**
   *Get the Max run number
   *Y. Guo April 20, 2007
   **/
  unsigned int DBWriter::maxRunNumber() {
    string sqlStmt = "select max(run_number) from "+  _dbOwner + ".CMS_RUNS";
    unsigned int maxRun;
    try {
      _stmt = _conn->createStatement (sqlStmt);
      ResultSet *rset = _stmt->executeQuery ();
      rset->next ();
      maxRun = rset->getUInt(1);
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
   }
   return maxRun;
  }
  
  /* 
   * June 11, 2007  Y Guo
   **/
  unsigned long DBWriter::getLumiSequence( const string& seqName) {
    string sel= "select " +  _dbOwner + "."+ seqName + ".nextval from dual";
    unsigned long seq_id;
    try {
      _stmt = _conn->createStatement (sel);
      ResultSet *rset = _stmt->executeQuery ();
      rset->next ();
      seq_id = (unsigned long)rset->getUInt(1);
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      std::string msg = seqName + "\n";
      msg += aExc.getMessage();
      OracleDBException lExc(msg);
      RAISE(lExc);
    }
    return seq_id;
  }

  /**
   * Get next sequence value from THRESHOLD_BITMASK_SEQ.
   **/
  unsigned long DBWriter::getThresholdConfigMId() {
    unsigned long ret;
    try {
      ret = getLumiSequence("LUMI_THSH_BMSK_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }

  /**
   * Get the sequence for lumi_tags table.
   **/ 
  unsigned long DBWriter::getLumiTagSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("LUMI_TAG_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }
  
  /**
   * Get next sequence value from TRG_DTIME_DEF_SEQ.
   **/
  unsigned long DBWriter::getTrgDefSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("TRG_DTIME_DEF_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }
  
  /**
   * Get next sequence value from L1_HLT_TRG_SEQ.
   **/
  unsigned long DBWriter::getL1TrgSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("L1_TRG_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }

  /**
   * Get next sequence value from TRG_DTIME_SEQ.
   **/
  unsigned long DBWriter::getTrgDtimeSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("TRG_DTIME_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }
  
  unsigned long DBWriter::getLumiSectionSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("LUMI_SECTION_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }

  unsigned long DBWriter::getCRunSeq() {
    unsigned long ret;
    try {
      ret = getLumiSequence("CRUN_SEQ");
    } catch (OracleDBException & aExc) {
      RETHROW(aExc);
    }
    return ret;
  }
  
  void DBWriter::insertCmsRuns(unsigned int runNum) {
    string sqlStmt = "insert into " + _dbOwner + ".cms_runs(Run_number) "
      + "VALUES (:runNum)";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, runNum);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

  unsigned long DBWriter::getHFRingSetId(unsigned int setVrsnNum) {
    string sel = "select ring_set_id from "+  _dbOwner + ".LUMI_HF_RING_SETS "
      "where set_version_number = :vNum ";
    unsigned long sdVrsid;
    ResultSet *rset;
    try {
      _stmt = _conn->createStatement(sel);
      _stmt->setUInt(1, setVrsnNum);   
      rset = _stmt->executeQuery ();
      while(rset->next ()) {
	sdVrsid = (unsigned long)rset->getUInt(1);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return sdVrsid;
  }
  
  void DBWriter::putTrgDeadtimeDefMap() {
    /*
     * Fill _deadtimeDefinitionMap
     *
     * Y. Guo  July 23, 07
     * Vervised Feb. 12, 08  Y.Guo
     *
     */
    string sel = "select trg_deadtime_name, trg_deadtime_def_id from "
      + _dbOwner
      + ".trigger_deadtime_defs";
    _stmt = _conn->createStatement (sel);
    
    ResultSet *rset; 
    try {
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	//cout<<"Name, Val:"<<rset->getString(1)<<"; "<<rset->getUInt(2)<<endl;
	_deadtimeDefinitionMap[rset->getString(1)]=(unsigned long)(rset->getUInt(2));
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for putTriggerDefinitionMap "<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  } //end putTrgDeadtimeDefMap()
  
  std::map<std::string, unsigned long> & DBWriter::getTrgDTDefMap() {
    /*
     * Return the _deadtimeDefinitionMap
     *
     * Y. Guo July 25, 07
     */
    return _deadtimeDefinitionMap;
  }

  /**
   * Function for bulk inseration to LEVEL1_HLT_TRIGGERS table
   *
   * Author Y. Guo July 25, 2007
   *
   */
  void DBWriter::insertArray_L1_trigger(unsigned long lsId,
					unsigned int* aL1LnNum, unsigned int* aL1Scl,
					unsigned int* aL1RtCnt, unsigned int* aHLTInt,
					unsigned int aLen) {
    int ntriger = 0;
    unsigned int NSEQ = 130; //this is the increament of the sequence. 
    int nseqFetch = 0;
    int nrem = 0;
    vector<unsigned long> seqSeed;
    if(aLen > NSEQ) {   
      ldiv_t q = div((long)aLen, (long)NSEQ);
      nrem = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      nrem = aLen;
    }
    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need. The sequence is incremented by 600 
      seqSeed.push_back(getL1TrgSeq());
    }
    
    //construct the sql
    string sql = "insert into "+ _dbOwner +"." + "LEVEL1_TRIGGERS"
      "(L1_TRG_ID, SECTION_ID, L1_LINE_NUMBER,"
      "L1_SCALER, L1_RATE_COUNTER,HLT_INPUT)"
      "VALUES (:tId, :lsId, :LNum, :LScl, :LRt,:HLTIPT)";
    //cout <<sql<<endl;
    //Fill Section ID
    ub2 aSecLen[aLen];
    unsigned long aSec[aLen];
    
    cout<<"section id = "<<lsId <<endl;
    for(unsigned int i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }

    //Fill trigger_id/sequence_id
    ub2 aSeqLen[aLen];
    unsigned long aSeq[aLen];
    unsigned int extra=NSEQ;
    for(vector<unsigned long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=nrem;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=(unsigned long)(seqSeed[j]+(i - j*NSEQ));
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }//end j loop
    //Fill for data
    ub2 aL1LnNumLen[aLen], aL1SclLen[aLen], aL1RtCntLen[aLen], aHLTIntLen[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aL1LnNumLen[i]=sizeof(aL1LnNum[i]);
      aL1SclLen[i]=sizeof(aL1Scl[i]);
      aL1RtCntLen[i]=sizeof(aL1RtCnt[i]);
      aHLTIntLen[i]=sizeof(aHLTInt[i]); 
    }
    _stmt=_conn->createStatement (sql);
    try {
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),
			   (unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]),
			   (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)aL1LnNum,  OCCIUNSIGNED_INT, sizeof(aL1LnNum[0]),
			   (unsigned short *)aL1LnNumLen);
      _stmt->setDataBuffer(4, (void*)aL1Scl, OCCIUNSIGNED_INT, sizeof(aL1Scl[0]),
                             (unsigned short *)aL1SclLen);
      _stmt->setDataBuffer(5, (void*)aL1RtCnt, OCCIUNSIGNED_INT, sizeof(aL1RtCnt[0]),
			   (unsigned short *)aL1RtCntLen);
      _stmt->setDataBuffer(6, (void*)aHLTInt, OCCIUNSIGNED_INT, sizeof(aHLTInt[0]),
			   (unsigned short *)aHLTIntLen);  
      //data insert for aLen rows
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for insertArray_L1HltTrg "<<endl;
      //cout<<sql<<endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  } //end DBWriter::insertArray_Trigger

  /**
   * The function for bulk inseration to TRIGGER_DEADTIMES table
   *
   * Original: Feb.12, 2008  Y. Guo
   *
   */
  void DBWriter::insertArray_deadtime(unsigned long lsId,
				      unsigned long* aDeadtimeDefId, 
				      unsigned int* aDeadtime,
				      unsigned int aLen) {
    int ntriger = 0;
    unsigned int NSEQ = 6; //this is the increament of the sequence.
    int nseqFetch = 0;
    int nrem = 0;
    vector<unsigned long> seqSeed;
    if(aLen > NSEQ) {
      ldiv_t q = div((long)aLen, (long)NSEQ);
      nrem = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      nrem = aLen;
    }
    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need. The sequence is incremented by 600
      seqSeed.push_back(getTrgDtimeSeq());
    }
    //construct the sql
    string sql = "insert into "+ _dbOwner +"." + "TRIGGER_DEADTIMES"
      "(TRG_DEADTIME_ID, SECTION_ID, TRG_DEADTIME_DEF_ID,TRG_DEADTIME)"+
      "VALUES (:tId, :lsId, :TDId, :TVal)";
    //Fill Section ID
    ub2 aSecLen[aLen];
    unsigned long  aSec[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    //Fill trigger_id/sequence_id
    ub2 aSeqLen[aLen];
    unsigned long aSeq[aLen];
    unsigned int extra=NSEQ;
    for(vector<unsigned long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=nrem;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }//end j loop
    //Fill for data
    ub2 aTrgDefIdLen[aLen], aTrgValLen[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aTrgDefIdLen[i]=sizeof(aDeadtimeDefId[i]);
      aTrgValLen[i]=sizeof(aDeadtime[i]);
    }
    try {
      _stmt=_conn->createStatement (sql);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),
			   (unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]),
			   (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)aDeadtimeDefId,  OCCIUNSIGNED_INT, sizeof(aDeadtimeDefId[0]), 
			   (unsigned short *)aTrgDefIdLen);
      _stmt->setDataBuffer(4, (void*)aDeadtime, OCCIINT, sizeof(aDeadtime[0]),
			   (unsigned short *)aTrgValLen);
      //data insert for aLen rows
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for insertArray_L1HltTrg "<<endl;
      //cout<<sql<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }//end DBWriter::insertArray_Deadtime       
  
  /*********************************************************************
   * Get section_id of a pair of run_number and lumi_section_number from 
   * lumi_sections table.  
   *
   * Y Guo September 10, 2007
   *
   *********************************************************************/ 
  unsigned long DBWriter::getLumiSecId(unsigned int runNum, unsigned int lsNum) {
    string sqlStmt = "select section_id  from "+ _dbOwner + ".lumi_sections where "
      "run_number=:rn and lumi_section_number=:ln";  
    ResultSet *rset;
    unsigned long lsId=0;
    try {
      _stmt = _conn->createStatement (sqlStmt);
      _stmt->setUInt(1, runNum);
      _stmt->setUInt(2, lsNum);
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	lsId=rset->getUInt(1);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return lsId;
  }

  /*********************************************************************
   * New signature for lumiSection table inseration. In the new db schema
   * threshod and bit mask are no longer required.
   *
   * Y Guo July 19, 2007
   *
   **********************************************************************/
  void DBWriter::insertBind_LumiSec(unsigned long lsId,
				    int HFRngStId,
				    int dataTaking,
				    unsigned int beginObt,
				    unsigned int totalObt,
				    unsigned int runNum,  
				    unsigned int lsNum,
				    unsigned int fillNum,
				    unsigned long lsStartT,
				    unsigned long lsStopT, 
				    const string & comment) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
      "RING_SET_ID, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER,"
      "FILL_NUMBER, SEC_START_TIME, SEC_STOP_TIME, COMMENTS)"
      "VALUES (:sid, :setVnum, :dataTk, :bO, :eO,:run, :lsNum,"
      ":fNum,:secStratT,:secStopT,:comm )";

    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, lsId);
      _stmt->setInt (2, HFRngStId);
      _stmt->setInt (3, dataTaking);
      _stmt->setUInt (4, beginObt);
      _stmt->setUInt (5, totalObt);
      _stmt->setUInt (6, runNum);
      _stmt->setUInt (7, lsNum);
      _stmt->setUInt (8, fillNum);
      _stmt->setUInt (9, lsStartT);
      _stmt->setUInt (10, lsStopT);
      _stmt->setString (11, comment);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /***********************************************************
   *
   * Insert into lumi_sections table with bare minimum info. 
   * Y. Guo Aug. 21 2007
   *
   ************************************************************/
  void DBWriter::insertBind_LumiSec(unsigned long lsId,
				    const DBWriter::DBLumiSection & summary) {
    //string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
    //"RING_SET_ID, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
    //"NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER)"
    //"VALUES (:sid, :stId, :dataTk, :bO, :eO,:run, :lsNum)";
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
      "IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER, RING_SET_ID)"
      "VALUES (:sid, :dataTk, :bO, :eO,:run, :lsNum, :rId)";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, lsId);
      _stmt->setUInt (2, summary.dataTaking);
      _stmt->setUInt (3, summary.beginObt);
      _stmt->setUInt (4, summary.totalObts);
      _stmt->setUInt (5, summary.runNum);
      _stmt->setUInt (6, summary.lsNum);
      _stmt->setUInt (7, summary.HFringStId);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /***********************************************************
   *
   * Insert into lumi_sections table with bare minimum info
   * + HF_RING_SET_ID that indicates what HF configuration 
   * was used for this lumi section.
   * 
   * Y. Guo Feb. 2008
   *
   ************************************************************/
  /*
  void DBWriter::insertBind_LumiSec(unsigned long lsId, unsigned long HFRngStId,
				    unsigned int dataTaking, unsigned int beginObt,
				    unsigned int totalObts,  unsigned int runNum,
				    unsigned int lsNum) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
      "RING_SET_ID, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER)"
      "VALUES (:sid, :stId, :dataTk, :bO, :eO,:run, :lsNum)";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, lsId);
      _stmt->setUInt (2, HFRngStId);
      _stmt->setUInt (3, dataTaking);
      _stmt->setUInt (4, beginObt);
      _stmt->setUInt (5, totalObts);
      _stmt->setUInt (6, runNum);
      _stmt->setUInt (7, lsNum);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    }*/

  /**********************************************************
   * Another signature for lumiSection table inseration
   * 
   * Y Guo April 18, 2007
   *
   **********************************************************/ 
  void DBWriter::insertBind_LumiSec(unsigned long lsId,
				    unsigned long thId,
				    unsigned long bitmId, 
				    int setVrsNum,
				    int dataTaking,
				    unsigned int beginObt, 
				    int unsigned totalObt,
				    unsigned int runNum,
				    unsigned int lsNum,
				    unsigned int fillNum,
				    unsigned long lsStartT,
				    unsigned long lsStopT, 
				    const string & comment) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID, BITMASK_ID," 
      "THRESHOLD_ID,RING_SET_ID, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER," 
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER," 
      "FILL_NUMBER, SEC_START_TIME, SEC_STOP_TIME, COMMENTS)"
      "VALUES (:sid, :bit ,:th,:setVnum, :dataTk, :bO, :eO,:run, :lsNum," 
      ":fNum,:secStratT,:secStopT,:comm )";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, lsId);
      _stmt->setUInt (2, thId);
      _stmt->setUInt (3, bitmId);
      _stmt->setInt (4, setVrsNum);
      _stmt->setInt (5, dataTaking);
      _stmt->setUInt (6, beginObt);
      _stmt->setUInt (7, totalObt);
      _stmt->setUInt (8, runNum);
      _stmt->setUInt (9, lsNum);
      _stmt->setUInt (10, fillNum);
      _stmt->setUInt (11, lsStartT);
      _stmt->setUInt (12, lsStopT);
      _stmt->setString (13, comment);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /*********************************************************************************
   * Insert into LUMI_DETAILS Table, that replace the old LUMI_ TABLE.
   *
   * January, 08  Y. Guo
   *
   *********************************************************************************/
  void DBWriter::insertArray_LumiDetails(unsigned long lsId,
					 const DBWriter::DBLumiDetails & details) {
    int Occ_len=0;
    int ntriger = 0;
    unsigned int NSEQ = 1000;
    int nseqFetch = 0;
    vector<unsigned long> seqSeed;
    
    if(details.aLen >NSEQ) {
      div_t q = div(details.aLen, (int)NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = details.aLen;
    }
    
    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need
      seqSeed.push_back(getLumiSequence("LUMI_DTLS_SEQ"));
    }
    //construct the sql
    string sqlStmt = "insert into " + _dbOwner + ".lumi_details(RECORD_ID,"
      "SECTION_ID, BUNCH_X_NUMBER, "
      "NORMALIZATION_ET, ET_LUMI, ET_LUMI_ERR, ET_LUMI_QLTY,"
      "NORMALIZATION_OCC_D1, OCC_LUMI_D1, OCC_LUMI_D1_ERR, OCC_LUMI_D1_QLTY,"
      "NORMALIZATION_OCC_D2, OCC_LUMI_D2, OCC_LUMI_D2_ERR, OCC_LUMI_D2_QLTY)"
      "VALUES(:rId, :sId, :bxNum, :NorEt, :etL, :etLE, :etLQ," 
      ":norOccD1, :occLD1, :occLD1E, :occLD1Q,"
      ":norOccD2, :occLD2, :occLD2E, :occLD2Q)";
    //Fill lumi section ID
    ub2 aSecLen[details.aLen];
    unsigned int aSec[details.aLen];
    for(unsigned int i=0; i<details.aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    //Fill record_id/sequence_id
    ub2 aSeqLen[details.aLen];
    unsigned long aSeq[details.aLen];
    int extra=NSEQ;
    for(vector<int>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }//end j loop
    
    //Fill for data
    ub2 aBxLen[details.aLen], aNorEtLumiLen[details.aLen], aEtLumiLen[details.aLen], aEtErrLen[details.aLen], aEtQLen[details.aLen];
    ub2 aNorOccLumiD1Len[details.aLen], aOccLumiD1Len[details.aLen];
    ub2 aOccLumiD1ErrLen[details.aLen], aOccLumiD1QLen[details.aLen];
    ub2 aNorOccLumiD2Len[details.aLen], aOccLumiD2Len[details.aLen];
    ub2 aOccLumiD2ErrLen[details.aLen], aOccLumiD2QLen[details.aLen];

    for(unsigned int i=0; i<details.aLen; i++) {
      aBxLen[i]=sizeof(details.aBX[i]);
      aNorEtLumiLen[i]=sizeof(details.aNorEtLumi[i]);
      aEtLumiLen[i]=sizeof(details.aEtLumi[i]);
      aEtErrLen[i]=sizeof(details.aEtLumiErr[i]);
      aEtQLen[i]=sizeof(details.aEtLumiQ[i]);
      aNorOccLumiD1Len[i]=sizeof(details.aNorOccLumiD1[i]);
      aOccLumiD1Len[i]=sizeof(details.aOccLumiLumiD1[i]);
      aOccLumiD1ErrLen[i]=sizeof(details.aOccLumiD1Err[i]);
      aOccLumiD1QLen[i]=sizeof(details.aOccLumiD1Q[i]);
      aNorOccLumiD2Len[i]=sizeof(details.aNorOccLumiD2[i]);
      aOccLumiD2Len[i]=sizeof(details.aOccLumiLumiD2[i]);
      aOccLumiD2ErrLen[i]=sizeof(details.aOccLumiD2Err[i]);
      aOccLumiD2QLen[i]=sizeof(details.aOccLumiD2Q[i]);
    }
    
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),(unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]), (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)details.aBX, OCCIUNSIGNED_INT, sizeof(details.aBX[0]), (unsigned short *)aBxLen);
      _stmt->setDataBuffer(4,(void*)details.aNorEtLumi, OCCIFLOAT, sizeof(details.aNorEtLumi[0]),
			   (unsigned short *)aNorEtLumiLen);
      _stmt->setDataBuffer(5,(void*)details.aEtLumi, OCCIFLOAT, sizeof(details.aEtLumi[0]),
			   (unsigned short *)aEtLumiLen);
      _stmt->setDataBuffer(6,(void*)details.aEtLumiErr, OCCIFLOAT, sizeof(details.aEtLumiErr[0]),
			   (unsigned short *)aEtErrLen);
      _stmt->setDataBuffer(7,(void*)details.aEtLumiQ, OCCIUNSIGNED_INT, sizeof(details.aEtLumiQ[0]),
			   (unsigned short *)aEtQLen);
      _stmt->setDataBuffer(8,(void*)details.aNorOccLumiD1, OCCIFLOAT, sizeof(details.aNorOccLumiD1[0]),
			   (unsigned short *)aNorOccLumiD1Len);
      _stmt->setDataBuffer(9,(void*)details.aOccLumiLumiD1, OCCIFLOAT, sizeof(details.aOccLumiLumiD1[0]),
			   (unsigned short *)aOccLumiD1Len);
      _stmt->setDataBuffer(10,(void*)details.aOccLumiD1Err, OCCIFLOAT, sizeof(details.aOccLumiD1Err[0]),
			   (unsigned short *)aOccLumiD1ErrLen);
      _stmt->setDataBuffer(11,(void*)details.aOccLumiD1Q, OCCIUNSIGNED_INT, sizeof(details.aOccLumiD1Q[0]),
			   (unsigned short *)aOccLumiD1QLen);
      _stmt->setDataBuffer(12,(void*)details.aNorOccLumiD2, OCCIFLOAT, sizeof(details.aNorOccLumiD2[0]),
			   (unsigned short *)aNorOccLumiD2Len);
      _stmt->setDataBuffer(13,(void*)details.aOccLumiLumiD2, OCCIFLOAT, sizeof(details.aOccLumiLumiD2[0]),
			   (unsigned short *)aOccLumiD2Len);
      _stmt->setDataBuffer(14,(void*)details.aOccLumiD2Err, OCCIFLOAT, sizeof(details.aOccLumiD2Err[0]),
			   (unsigned short *)aOccLumiD2ErrLen);
      _stmt->setDataBuffer(15,(void*)details.aOccLumiD2Q, OCCIUNSIGNED_INT, sizeof(details.aOccLumiD2Q[0]),
			   (unsigned short *)aOccLumiD2QLen);
      
      //data insert for aLen rows
      _stmt->executeArrayUpdate(details.aLen);
      _conn->terminateStatement(_stmt); 
    } catch(SQLException & aExc) {
      //cout<<sqlStmt<<endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

  /**************************************************************************
   *
   * Insert into Lumi_Summary Table. 
   *
   * January 09, 08   Y. Guo
   *
   **************************************************************************/
  void DBWriter::insertBind_LumiSummary(unsigned long lsId,
					const DBWriter::DBLumiSummary & summary) {
    string sqlStmt = "insert into "+ _dbOwner + ".LUMI_SUMMARIES(SECTION_ID,"
      "DEADTIME_NORMALIZATION,"
      "NORMALIZATION, INSTANT_LUMI, INSTANT_LUMI_ERR, INSTANT_LUMI_QLTY,"
      "NORMALIZATION_ET, INSTANT_ET_LUMI, INSTANT_ET_LUMI_ERR,"
      "INSTANT_ET_LUMI_QLTY, NORMALIZATION_OCC_D1, INSTANT_OCC_LUMI_D1," 
      "INSTANT_OCC_LUMI_D1_ERR,INSTANT_OCC_LUMI_D1_QLTY,"
      "NORMALIZATION_OCC_D2, INSTANT_OCC_LUMI_D2, INSTANT_OCC_LUMI_D2_ERR,"
      "INSTANT_OCC_LUMI_D2_QLTY) VALUES(:sId, :deadT, :nor, :Lumi,"
      ":LumiErr, :LumiQ, :norEt, :EtLumi, :EtErr, :EtQ, :norOccD1,:OccLumiD1, "
      ":OccD1Err, :OccD1Q, :norOccD2, :OccLumiD2, :OccD2Err, :OccD2Q)";
    
    _stmt=_conn->createStatement (sqlStmt);
    try {
      _stmt->setUInt(1, lsId);
      _stmt->setFloat(2, summary.dTimeNorm);
      _stmt->setFloat(3, summary.norm);
      _stmt->setFloat(4, summary.instLumi);
      _stmt->setFloat(5, summary.instLumiErr);
      _stmt->setUInt(6, summary.instLumiQ);
      _stmt->setFloat(7, summary.norEt);
      _stmt->setFloat(8,summary.instEtLumi );
      _stmt->setFloat(9,summary.instEtLumiErr );
      _stmt->setUInt(10, summary.instEtLumiQ);
      _stmt->setFloat(11,summary.norOccLumiD1);
      _stmt->setFloat(12,summary.instOccLumiD1);
      _stmt->setFloat(13,summary.instOccLumiD1Err);
      _stmt->setUInt(14, summary.instOccLumiD1Q);
      _stmt->setFloat(15,summary.norOccLumiD2);
      _stmt->setFloat(16,summary.instOccLumiD2);
      _stmt->setFloat(17,summary.instOccLumiD2Err);
      _stmt->setUInt(18, summary.instOccLumiD2Q);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /****************************************************************************************
   * Get lumi_summary_id from lumi_summarys table.
   *
   * Y. Guo September 11 2007
   *
   **/
  unsigned long DBWriter::getLumiSummaryId(unsigned long lsId, int lumiVer) {
    string sqlStmt = "select lumi_summary_id from " + _dbOwner + ".lumi_summaries where "
      "section_id=:sId and lumi_version=:lv";
    ResultSet *rset;
    unsigned long smId=0;
    try {
      _stmt = _conn->createStatement (sqlStmt);
      _stmt->setUInt(1, lsId);
      _stmt->setInt(2, lumiVer);
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	smId=(unsigned long)rset->getUInt(1);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    //_stmt->closeResultSet (rset);
    //_conn->terminateStatement(_stmt);
    return smId;
  }
  
  /**********************************************************************************
   * Create a lumi tag named tName.
   *
   * Y Guo September 17, 07
   *
   **/
  unsigned long DBWriter::insertLumiTag(std::string& tName) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_tags(lumi_tag_id, tag_name) "
      "values (:tId, :tN)";
    unsigned long id;
    try {
      id = getLumiTagSeq();
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt(1, id);
      _stmt->setString(2, tName);
      _stmt->executeUpdate ();
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return id;
  }

  /***********************************************************************************
   *Get lumi_tag_id of TAG_NAME tName from lumi_tags table.
   * 
   * Y Guo September 10, 07
   *
   **/
  unsigned long DBWriter::getLumiTagId(std::string& tName) {
    string sqlStmt = "select lumi_tag_id  from " + _dbOwner + ".lumi_tags where "
      "tag_name=:tn";
    ResultSet *rset;
    unsigned long lsId=0;
    try {
      _stmt = _conn->createStatement (sqlStmt);
      _stmt->setString(1, tName);
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	lsId=rset->getUInt(1);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException & aExc) {
      //_stmt->closeResultSet (rset);
      //_conn->terminateStatement(_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return lsId;
  }
  
  /*************************************************************************************
   *
   * insert into lumi_version_tag_maps table. The API finds out the lumi_summary_id that 
   * assocats with the lumi_section and lumi_version, lumi_tag_id accocated with the 
   * tag_name then inserts them into the map.
   *
   * Y. Guo September 11 07
   *
   **************************************************************************************/
  void DBWriter::insertLumiVerTagMap(unsigned int lsNum, unsigned int runNum, 
				     int lumiVer, string& tName) {
    unsigned long lsId, tagId, summaryId;
    try {
      lsId = getLumiSecId(runNum, lsNum);
      tagId = getLumiTagId(tName);
      summaryId = getLumiSummaryId(lsId, lumiVer);
      string sqlStmt = "insert into " + _dbOwner + ".lumi_version_tag_maps(lumi_tag_id, "
	"lumi_summary_id) values (:tId, :sId)"; 
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt(1, tagId);
      _stmt->setUInt(2, summaryId);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**
   * Insert into HLT table.
   *
   * Y. Guo Feb.22 2008
   *
   **/
  //This API still under develoment. YG
  void DBWriter::insertArray_HLTs(unsigned long lsId, char* aTrgPth,
                          unsigned long* aInptCnt, unsigned long* aAccptCnt,
                          unsigned int* aPreSclFct, unsigned int aLen) {
    int hlt_len=0;
    int ntriger = 0;
    unsigned int NSEQ = 20;  //sequence increase step 
    int nseqFetch = 0;
    vector<unsigned long> seqSeed;
    if(aLen > NSEQ) {
      div_t q = div(aLen, (int)NSEQ);
      hlt_len = q.rem; //the remainder
      ntriger = q.quot; 
    } else {
      hlt_len = aLen; 
    }

    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need
      seqSeed.push_back(getLumiSequence("HLT_SEQ" ));
    }
    
    // construct the sql
    string sqlStmt = "insert into " + _dbOwner + ".HLTS(HLT_ID,"
      "SECTION_ID, TRIGGER_PATH, "
      "INPUT_COUNT, ACCEPT_COUNT, PRESCALE_FACTOR)"
      "VALUES(:rId, :sId, :tPath, :iCnt, :aCnt, :pFct)";

    // Fill lumi section ID
    ub2 aSecLen[aLen];
    unsigned int aSec[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    //Fill record_id/sequence_id
    ub2 aSeqLen[aLen];
    unsigned long aSeq[aLen];
    int extra=NSEQ;
    for(vector<int>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=hlt_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    } // end j loop

    // Fill for data
    ub2  aInptCntLen[aLen], aTrgPthLen[aLen], aAccptCntLen[aLen],  aPreSclFctLen[aLen]; 
    for(unsigned int i=0; i<aLen; i++) {
      aInptCntLen[i]=sizeof(aInptCnt[i]);
      aTrgPthLen[i]=3;
      aAccptCntLen[i]=sizeof(aAccptCnt[i]);
      aPreSclFctLen[i]=sizeof(aPreSclFct[i]);
    }

    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),(unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]), (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)aTrgPth, OCCI_SQLT_STR, 3, (unsigned short *)aTrgPthLen);
      _stmt->setDataBuffer(4,(void*)aInptCnt, OCCIUNSIGNED_INT, sizeof(aInptCnt[0]),
			   (unsigned short *)aInptCntLen);
      _stmt->setDataBuffer(5,(void*)aAccptCnt, OCCIUNSIGNED_INT, sizeof(aAccptCnt[0]),
			   (unsigned short *)aAccptCntLen);
      _stmt->setDataBuffer(6,(void*)aPreSclFct, OCCIUNSIGNED_INT, sizeof(aPreSclFct[0]),
			   (unsigned short *)aPreSclFctLen);
      //data insert for aLen rows
      //cout << "insert now . "<<endl;
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for insertArray_HLTS"<<endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

  /**
   * Insertion into _dbOwner.lumi_thresholds table. 
   * Db Trigger for threshold_id is not used since we need the id for lumi_ssection
   * table right way.
   * Y. Guo June 8 2007
   **/
  void DBWriter::insertBind_threshold (unsigned long thId, int th1Set1, int th1Set2, int th2Set1,
				       int th2Set2, int thEt) {
    string sqlStmt = "insert into " + _dbOwner +".lumi_thresholds(THRESHOLD_ID,"
      " THRESHOLD1_SET1, THRESHOLD1_SET2, THRESHOLD2_SET1,"
      " THRESHOLD2_SET2, ET_THRESHOLD)VALUES (:tId, :t1S1, :t1S2,"
      " :t2S1, :t2S2, :te)";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, thId);
      _stmt->setInt (2, th1Set1);
      _stmt->setInt (3, th1Set2);
      _stmt->setInt (4, th2Set1);
      _stmt->setInt (5, th2Set2);
      _stmt->setInt (6, thEt);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for insertBind"<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      // _conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    //_conn->terminateStatement (_stmt);
  }
  
  /**
   * Insert Configuration_BITMASKS table. Trigger is not used for BOTMASK_ID because  
   * we will need the id to fill the lumi_section table.
   * June 8, 2007  Y. Guo
   **/
  void DBWriter::insertBind_ConfigBitMask (const long bitMskId,
					   const int mskForward[18],
					   const int mskBackward[18]) {
    string sqlStmt = "insert into "+ _dbOwner +".configuration_bitmasks(BITMASK_ID,"
      "FORWARD_BITMASK1, FORWARD_BITMASK2, FORWARD_BITMASK3,"
      "FORWARD_BITMASK4, FORWARD_BITMASK5, FORWARD_BITMASK6,"
      "FORWARD_BITMASK7, FORWARD_BITMASK8, FORWARD_BITMASK9,"
      "FORWARD_BITMASK10, FORWARD_BITMASK11, FORWARD_BITMASK12,"
      "FORWARD_BITMASK13, FORWARD_BITMASK14, FORWARD_BITMASK15,"
      "FORWARD_BITMASK16, FORWARD_BITMASK17, FORWARD_BITMASK18,"
      "BACKWARD_BITMASK1, BACKWARD_BITMASK2, BACKWARD_BITMASK3,"
      "BACKWARD_BITMASK4, BACKWARD_BITMASK5, BACKWARD_BITMASK6,"
      "BACKWARD_BITMASK7, BACKWARD_BITMASK8, BACKWARD_BITMASK9,"
      "BACKWARD_BITMASK10, BACKWARD_BITMASK11, BACKWARD_BITMASK12,"
      "BACKWARD_BITMASK13, BACKWARD_BITMASK14, BACKWARD_BITMASK15,"
      "BACKWARD_BITMASK16, BACKWARD_BITMASK17, BACKWARD_BITMASK18)"
      "VALUES (:bmId, :fBM1, :fBM2, :fBM3, :fBM4, :fBM5, :fBM6, :fBM7,"
      ":fBM8, :fBM9, :fBM10, :fBM11, :fBM12, :fBM13, :fBM14, :fBM15, :fBM16,"
      ":fBM17, :fBM18, :bBM1, :bBM2, :bBM3, :bBM4, :bBM5, :bBM6, :bBM7,"
      ":bBM8, :bBM9, :bBM10, :bBM11, :bBM12, :bBM13, :bBM14, :bBM15, :bBM16,"
      ":bBM17, :bBM18)";
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setUInt (1, bitMskId);
      _stmt->setInt (2, mskForward[0]);      _stmt->setInt (20, mskBackward[0]);
      _stmt->setInt (3, mskForward[1]);      _stmt->setInt (21, mskBackward[1]);
      _stmt->setInt (4, mskForward[2]);      _stmt->setInt (22, mskBackward[2]);
      _stmt->setInt (5, mskForward[3]);      _stmt->setInt (23, mskBackward[3]);
      _stmt->setInt (6, mskForward[4]);      _stmt->setInt (24, mskBackward[4]);
      _stmt->setInt (7, mskForward[5]);      _stmt->setInt (25, mskBackward[5]);
      _stmt->setInt (8, mskForward[6]);      _stmt->setInt (26, mskBackward[6]);
      _stmt->setInt (9, mskForward[7]);      _stmt->setInt (27, mskBackward[7]);
      _stmt->setInt (10, mskForward[8]);     _stmt->setInt (28, mskBackward[8]);
      _stmt->setInt (11, mskForward[9]);    _stmt->setInt (29, mskBackward[9]);
      _stmt->setInt (12, mskForward[10]);    _stmt->setInt (30, mskBackward[10]);
      _stmt->setInt (13, mskForward[11]);    _stmt->setInt (31, mskBackward[11]);
      _stmt->setInt (14, mskForward[12]);    _stmt->setInt (32, mskBackward[12]);
      _stmt->setInt (15, mskForward[13]);    _stmt->setInt (33, mskBackward[13]);
      _stmt->setInt (16, mskForward[14]);    _stmt->setInt (34, mskBackward[14]);
      _stmt->setInt (17, mskForward[15]);    _stmt->setInt (35, mskBackward[15]);
      _stmt->setInt (18, mskForward[16]);    _stmt->setInt (36, mskBackward[16]);
      _stmt->setInt (19, mskForward[17]);    _stmt->setInt (37, mskBackward[17]);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for inserting configuration_bitmask table. "<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    //_conn->terminateStatement (_stmt);
  }

  /**
   * Function for  bulk bind inseration to all the OCC histogram tables 
   * 
   * Author Y. Guo June 21, 2006
   *
   */
  
  void DBWriter::insertArray_allOccHist(unsigned long lsId, int* aBX, unsigned long* aOcc,
					unsigned long* aLostLnb,int* aHlxNum,
					int* aRingSetNum, unsigned long aLen,
					const string& tableName) {
    int Occ_len=0;
    int ntriger = 0;
    int NSEQ = 1000;
    int nseqFetch = 0;
    vector<unsigned long> seqSeed;

    if(aLen > abs(NSEQ)) {
      ldiv_t q = div(aLen, (long)NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = (int)aLen;
    }
    
    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need. The sequence is incremented by 4000
      seqSeed.push_back(getLumiSequence("lumi_all_histgram_seq"));
    }
    
    // construct the sql
    string sqlStmt = "insert into "+ _dbOwner +"." + tableName +
      "(RECORD_ID, SECTION_ID, BUNCH_X_NUMBER,TOTAL_OCCUPANCIES,"
      "HLX_NUMBER, RING_SET_NUMBER, LOST_LNB_COUNT)"
      "VALUES (:rId, :lsId, :bxNum, :occ ,:hlxNum, :rSetNum, :lstLnbCnt)";
    // Fill Section ID
    ub2 aSecLen[aLen];
    unsigned long aSec[aLen];
    for(unsigned long i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    // Fill record_id/sequence_id
    ub2 aSeqLen[aLen];
    unsigned long aSeq[aLen];
    int extra=NSEQ;
    for(vector<unsigned long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    } // end j loop
    
    //Fill for data
    ub2 aBxLen[aLen], aOccLen[aLen], aLostLnbLen[aLen], aHlxNumLen[aLen];
    ub2 aRingSetNumLen[aLen];
    for(unsigned long i=0; i<aLen; i++) {
      aBxLen[i]=sizeof(aBX[i]);
      aOccLen[i]=sizeof(aOcc[i]);
      aLostLnbLen[i]=sizeof(aLostLnb[i]);
      aHlxNumLen[i]=sizeof(aHlxNum[i]);
      aRingSetNumLen[i]=sizeof(aRingSetNum[i]);
    }
    
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),
			   (unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]),
			   (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)(aBX),  OCCIINT, sizeof(aBX[0]),
			   (unsigned short *)aBxLen);
      _stmt->setDataBuffer(4,(void*)(aOcc), OCCIINT, sizeof(aOcc[0]),
			   (unsigned short *)aOccLen);
      _stmt->setDataBuffer(7,(void*)(aLostLnb), OCCIINT, sizeof(aLostLnb[0]),
			   (unsigned short *)aLostLnbLen);
      _stmt->setDataBuffer(5,(void*)(aHlxNum), OCCIINT, sizeof(aHlxNum[0]),
			   (unsigned short *)aHlxNumLen);
      _stmt->setDataBuffer(6,(void*)(aRingSetNum), OCCIINT, sizeof(aRingSetNum[0]),
			   (unsigned short *)aRingSetNumLen);
      // data insert for aLen rows
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch(SQLException & aExc) {
      //cout<<"Exception thrown for insertArray_allOccHist "<<endl;
      //cout<<sqlStmt<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    // _conn->terminateStatement (_stmt);
  } // done insertArray_allHist  

  /***********************************************************
   * 
   * Insert into ET_SUM table
   * Y. Guo    June 8, 2007
   *
   **********************************************************/
  void DBWriter::insertArray_EtSum(unsigned long secid, int* aBX, double* aEt,
				   int* aHlxNum, long* aLostLnbCt,
				   unsigned long aLen) {
    int Occ_len=0;
    int ntriger = 0;
    int NSEQ = 1000;
    int nseqFetch = 0;
    vector<unsigned long> seqSeed;
    
    if(aLen >abs(NSEQ)) {
      ldiv_t q = div(aLen, (long)NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = (int)aLen;
    }
    
    //total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      //get all the sequence seeds we need
      long myrecord = getLumiSequence("lumi_all_histgram_seq");
      //cout << "ET_SUM record : " <<myrecord<<endl;
      seqSeed.push_back(myrecord);
    }
    //construct the sql
    
    string sqlStmt = "insert into "+ _dbOwner+ ".ET_SUMS"
      "(record_id, section_id,bunch_x_number,"
      " ET, HLX_NUMBER, LOST_LNB_COUNT)"
      " VALUES (:rId, :lsId, :bxnum, :et, :hlxNum, :lstLnbCt)";
    //Fill Section ID
    ub2 aSecLen[aLen];
    unsigned long aSec[aLen];
    for(unsigned long i=0; i<aLen; i++) {
      aSec[i]=secid;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(secid);
    }
    
    //Fill record_id/sequence_id
    ub2 aSeqLen[aLen];
    unsigned long aSeq[aLen];
    int extra=NSEQ;
    for(vector<int>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    } // end j loop
    
    // Fill for data
    ub2 aBxLen[aLen], aEtLen[aLen], aLostLnbCtLen[aLen], aHlxNumLen[aLen];
    for(unsigned long i=0; i<aLen; i++){
      aBxLen[i]=sizeof(aBX[i]);
      aEtLen[i]=sizeof(aEt[i]);
      aLostLnbCtLen[i]=sizeof(aLostLnbCt[i]);
      aHlxNumLen[i]=sizeof(aHlxNum[i]);
    }
    
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIUNSIGNED_INT, sizeof(aSeq[0]),(unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIUNSIGNED_INT, sizeof(aSec[0]), (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)(aBX),  OCCIINT, sizeof(aBX[0]),  (unsigned short *)aBxLen);
      _stmt->setDataBuffer(4,(void*)(aEt), OCCIFLOAT, sizeof(aEt[0]), (unsigned short *)aEtLen);
      _stmt->setDataBuffer(5,(void*)(aHlxNum), OCCIINT, sizeof(aHlxNum[0]),
			   (unsigned short *)aHlxNumLen);
      _stmt->setDataBuffer(6,(void*)(aLostLnbCt), OCCIINT, sizeof(aLostLnbCt[0]),
			   (unsigned short *)aLostLnbCtLen);
      
      //data insert for aLen rows
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
      //cout << "insert - Success" << endl;
    }catch(SQLException & aExc){
      //cout<<"Exception thrown for insertion for ET_SUM "<<endl;
      //cout<<sqlStmt<<endl;
      //cout<<"Error number: "<<  ex.getErrorCode() << endl;
      //cout<<ex.getMessage() << endl;
      //_conn->terminateStatement (_stmt);
      //throw ex;
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    //_conn->terminateStatement (_stmt);
  }// insertArray_allHist
  
}
