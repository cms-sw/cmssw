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
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  DBWriter::~DBWriter () {
    try {
      if ( _conn ) {
	_env->terminateConnection (_conn);
	_conn = 0;
      }
      if ( _env ) {
	Environment::terminateEnvironment (_env);
	_env = 0;
      }
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**
   *Get the Max run number
   *Y. Guo April 20, 2007
   **/
  int DBWriter::maxRunNumber() {
    string sqlStmt = "select max(run_number) from "+  _dbOwner + ".lumi_sections";
    int maxRun;
    try {
      _stmt = _conn->createStatement (sqlStmt);
      ResultSet *rset = _stmt->executeQuery ();
      rset->next ();
      maxRun = (int)rset->getInt(1);
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch(SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return maxRun;
  }
  
  /**
   * Get next sequence value from THRESHOLD_BITMASK_SEQ.
   **/
  long DBWriter::getThresholdConfigMId() {
    long retVal;
    try {
      retVal = getLumiSequence("LUMI_THSH_BMSK_SEQ");
    } catch (OracleDBException aExc) {
      RETHROW(aExc);
    }
    return retVal;
  }
  
  /**
   * Get next sequence value from TRG_DTIME_DEF_SEQ.
   **/
  long DBWriter::getTrgDefSeq() {
    long retVal;
    try {
      retVal = getLumiSequence("TRG_DTIME_DEF_SEQ");
    } catch (OracleDBException aExc) {
      RETHROW(aExc);
    }
    return retVal;
  }
  
  /**
   * Get next sequence value from L1_HLT_TRG_SEQ.
   **/
  long DBWriter::getTrgSeq() {
    long retVal;
    try {
      retVal = getLumiSequence("L1_HLT_TRG_SEQ");
    } catch (OracleDBException aExc) {
      RETHROW(aExc);
    }
    return retVal;
  }
  
  /**
   * Get next sequence value from TRG_DTIME_SEQ.
   **/
  long DBWriter::getTrgDtimeSeq() {
    long retVal;
    try {
      retVal = getLumiSequence("TRG_DTIME_SEQ");
    } catch (OracleDBException aExc) {
      RETHROW(aExc);
    }
    return retVal;
  }
  
  /* 
   * June 11, 2007  Y Guo
   **/
  long DBWriter::getLumiSequence( const string& seqName) {
    string sel= "select " +  _dbOwner + "."+ seqName + ".nextval from dual";
    long seq_id;
    try {
      _stmt = _conn->createStatement (sel);
      ResultSet *rset = _stmt->executeQuery ();
      rset->next ();
      seq_id = (long)rset->getInt(1);
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    return seq_id;
  }
  
  long DBWriter::getLumiSectionSeq() {
    long retVal;
    try {
      retVal = getLumiSequence("lumi_section_seq");
    } catch (OracleDBException aExc) {
      RETHROW(aExc);
    }
    return retVal;
  }
  
  /*
   * Fill _triggerDefinitionMap and _deadtimeDefinitionMap
   *
   * Y. Guo  July 23, 07
   *
   */
  /*void DBWriter::putTriggerDeadtimeDefinitionMap(int isDeadtime) {
    string sel = "select trigger_def_name, trigger_def_id from "
      + _dbOwner
      + ".trg_and_deadtime_definitions where is_deadtime = :ist";
    ResultSet *rset; 
    try {
      _stmt = _conn->createStatement (sel);
      _stmt->setInt(1, isDeadtime);
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	if(isDeadtime==0)_triggerDefinitionMap[rset->getString(1)]=rset->getInt(2);
	if(isDeadtime==1)_deadtimeDefinitionMap[rset->getString(1)]=rset->getInt(2);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
    }*/
  
  /*
   * Fill _triggerDefinitionMap
   * 
   * Y. Guo  July 23, 07
   *
   */
  void DBWriter::putTriggerDefinitionMap() {
    string sel = "select trigger_def_name, trigger_def_id from "
      + _dbOwner
      + ".trg_and_deadtime_definitions where is_deadtime = :ist";
    ResultSet *rset; 
    try {
      _stmt = _conn->createStatement (sel);
      _stmt->setInt(1, 0);  // 0 means is_deadtime == 0
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	_triggerDefinitionMap[rset->getString(1)]=rset->getInt(2);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

  /*
   * Fill _deadtimeDefinitionMap
   *
   * Y. Guo  July 23, 07
   *
   */
  void DBWriter::putDeadtimeDefinitionMap() {
    string sel = "select trigger_def_name, trigger_def_id from "
      + _dbOwner
      + ".trg_and_deadtime_definitions where is_deadtime = :ist";
    ResultSet *rset; 
    try {
      _stmt = _conn->createStatement (sel);
      _stmt->setInt(1, 0);  // 1 means is_deadtime == 1
      rset = _stmt->executeQuery ();
      while(rset->next()) {
	_deadtimeDefinitionMap[rset->getString(1)]=rset->getInt(2);
      }
      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  const std::map<std::string, int> & DBWriter::getDeadtimeDefinitionMap()
  {
    /*
     * Return the _deadtimeDefinitionMap
     *
     * Y. Guo July 25, 07
     */
    return _deadtimeDefinitionMap;
  }
  
  const std::map<std::string, int> & DBWriter::getTriggerDefinitionMap()
  {
    /*
     * Return the _triggerDefinitionMap
     *
     * Y. Guo July 25, 07
     */
    return _triggerDefinitionMap;
  }

  /**
   * Function for bulk inseration to LEVEL1_HLT_TRIGGERS table
   *
   * Author Y. Guo July 25, 2007
   *
   */
  void DBWriter::insertArray_trigger(long lsId, int* aTrgDefId, int* aTrgBitNum,
				    int* aTrgVal, unsigned int aLen) {
    int ntriger = 0;
    unsigned int NSEQ = 600; //this is the increment of the sequence. 
    int nseqFetch = 0;
    int nrem = 0;
    vector<long> seqSeed;
    if(aLen > NSEQ) {   
      ldiv_t q = div((long)aLen, (long)NSEQ);
      nrem = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      nrem = aLen;
    }

    // Total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i = 0; i < nseqFetch; i++) {
      //get all the sequence seeds we need. The sequence is incremented by 600 
      //long mysqc = getTrgSeq();
      //cout << mysqc <<endl; 
      seqSeed.push_back(getTrgSeq());
    }
    
    // Construct the sql
    string sql = "insert into "+ _dbOwner +"." + "LEVEL1_HLT_TRIGGERS" +
      "(TRIGGER_ID, SECTION_ID, TRIGGER_DEF_ID,TRIGGER_VALUE,"
      "TRIGGER_BIT_NUMBER)"
      "VALUES (:tId, :lsId, :TDId, :TVal, :TBNum)";

    // Fill Section ID
    ub2 aSecLen[aLen];
    long aSec[aLen];
    
    cout << "section id = " << lsId << endl;
    for(unsigned int i = 0; i < aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    // Fill trigger_id/sequence_id
    ub2 aSeqLen[aLen];
    long aSeq[aLen];
    unsigned int extra=NSEQ;
    for( vector<long>::size_type j=0; j < abs(nseqFetch); j++ ) {
      if ( j == abs(nseqFetch - 1) ) extra = nrem;
      for( unsigned int i = j * NSEQ ; i < (j * NSEQ + extra) ; i++ ) {
	aSeq[i] = seqSeed[j] + (i - j * NSEQ);
	aSeqLen[i] = sizeof(aSeq[i]);
      }
    }

    // Fill for data
    ub2 aTrgDefIdLen[aLen], aTrgBitNumLen[aLen], aTrgValLen[aLen];
    for(unsigned int i = 0; i < aLen; i++) {
	aTrgDefIdLen[i]=sizeof(aTrgDefId[i]);
	aTrgBitNumLen[i]=sizeof(aTrgBitNum[i]);
	aTrgValLen[i]=sizeof(aTrgVal[i]); 
    }     

    try {
      _stmt=_conn->createStatement (sql);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIINT, sizeof(aSeq[0]),
			   (unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIINT, sizeof(aSec[0]),
			   (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)aTrgDefId,  OCCIINT, sizeof(aTrgDefId[0]),
			   (unsigned short *)aTrgDefIdLen);
      _stmt->setDataBuffer(4, (void*)aTrgVal, OCCIINT, sizeof(aTrgVal[0]),
			   (unsigned short *)aTrgValLen);
      _stmt->setDataBuffer(5, (void*)aTrgBitNum, OCCIINT, sizeof(aTrgBitNum[0]),
			   (unsigned short *)aTrgBitNumLen);  
      // data insert for aLen rows
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**
   * Another function for bulk inseration to LEVEL1_HLT_TRIGGERS table
   *
   * Author Y. Guo July 25, 2007
   *
   */
  void DBWriter::insertArray_trigger(long lsId,
				    int *aTrgDefId, 
				    int *aTrgVal,
				    unsigned int aLen) {
    int ntriger = 0;
    unsigned int NSEQ = 600; //this is the increament of the sequence.
    int nseqFetch = 0;
    int nrem = 0;
    vector<long> seqSeed;
    if(aLen > NSEQ) {
      ldiv_t q = div((long)aLen, (long)NSEQ);
      nrem = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      nrem = aLen;
    }

    // total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i = 0 ; i < nseqFetch ; i++) {
      // get all the sequence seeds we need. The sequence is incremented by 600
      seqSeed.push_back(getTrgSeq());
    }

    // construct the sql
    string sql = "insert into "+ _dbOwner +"." + "LEVEL1_HLT_TRIGGERS" +
      "(TRIGGER_ID, SECTION_ID, TRIGGER_DEF_ID,TRIGGER_VALUE)"+
      "VALUES (:tId, :lsId, :TDId, :TVal)";
    // Fill Section ID
    ub2 aSecLen[aLen];
    long aSec[aLen];
    for(unsigned int i = 0; i < aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    // Fill trigger_id/sequence_id
    ub2 aSeqLen[aLen];
    long aSeq[aLen];
    unsigned int extra=NSEQ;
    for(vector<long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=nrem;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }

    // Fill for data
    ub2 aTrgDefIdLen[aLen], aTrgValLen[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aTrgDefIdLen[i]=sizeof(aTrgDefId[i]);
      aTrgValLen[i]=sizeof(aTrgVal[i]);
    }

    try {
	_stmt=_conn->createStatement (sql);
	_stmt->setDataBuffer(1,(void*)aSeq, OCCIINT, sizeof(aSeq[0]),
			     (unsigned short *)aSeqLen);
	_stmt->setDataBuffer(2,(void*)aSec, OCCIINT, sizeof(aSec[0]),
			     (unsigned short *)aSecLen);
	_stmt->setDataBuffer(3,(void*)aTrgDefId,  OCCIINT, sizeof(aTrgDefId[0]), 
                             (unsigned short *)aTrgDefIdLen);
	_stmt->setDataBuffer(4, (void*)aTrgVal, OCCIINT, sizeof(aTrgVal[0]),
                             (unsigned short *)aTrgValLen);
	// data insert for aLen rows
	_stmt->executeArrayUpdate(aLen);
	_conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**
   * Funtion to insert array of deadtime to table _dbOwner.trigger_deadtimes
   * Author: Y. Guo  July 25, 2007
   *
   **/ 
  void DBWriter::insertArray_deadtime(long lsId,
				     int * aDeadtimeDef, 
				     long * aDeadtime,
				     unsigned int aLen ) {
    int ntriger = 0;
    unsigned int NSEQ = 14; //this is the increament of the sequence.
    int nseqFetch = 0;
    int nrem = 0;
    vector<long> seqSeed;

    if(aLen > NSEQ) {
      ldiv_t q = div((long)aLen, (long)NSEQ);
      nrem = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      nrem = aLen;
    }
    
    // total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      // get all the sequence seeds we need. The sequence is incremented by 600
      seqSeed.push_back(getTrgSeq());
    }

    // construct the sql
    string sql = "insert into "+ _dbOwner +"." + "TRIGGER_DEADTIMES" +
      "(TRG_DEADTIME_ID, SECTION_ID, TRIGGER_DEF_ID,TRG_DEADTIME)"+
      "VALUES (:tId, :lsId, :TDId, :TDeadtime)";

    // Fill Section ID
    ub2 aSecLen[aLen];
    long aSec[aLen];
    
    for(unsigned int i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }

    // Fill trg__deadtime_id/sequence_id
    ub2 aSeqLen[aLen];
    long aSeq[aLen];
    unsigned int extra=NSEQ;

    for(vector<long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=nrem;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }

    // Fill for data
    ub2 aDeadtimeDefLen[aLen], aDeadtimeLen[aLen];
    for(unsigned int i=0; i<aLen; i++) {
      aDeadtimeDefLen[i]=sizeof(aDeadtimeDef[i]);
      aDeadtimeLen[i]=sizeof(aDeadtime[i]);
    }

    try {
	_stmt=_conn->createStatement (sql);
	_stmt->setDataBuffer(1,(void*)aSeq, OCCIINT, sizeof(aSeq[0]),
			     (unsigned short *)aSeqLen);
	_stmt->setDataBuffer(2,(void*)aSec, OCCIINT, sizeof(aSec[0]),
			     (unsigned short *)aSecLen);
	_stmt->setDataBuffer(3,(void*)aDeadtimeDef,  OCCIINT, sizeof(aDeadtimeDef[0]),
                             (unsigned short *)aDeadtimeDefLen);
	_stmt->setDataBuffer(4, (void*)aDeadtime, OCCIINT, sizeof(aDeadtime[0]),
                             (unsigned short *)aDeadtimeLen);
	// data insert for aLen rows
	_stmt->executeArrayUpdate(aLen);
	_conn->terminateStatement (_stmt);   
    } catch (SQLException aExc) {
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
  void DBWriter::insertBind_threshold (long thId,
				      int th1Set1,
				      int th1Set2,
				      int th2Set1, 
				      int th2Set2,
				      int thEt) {
    string sqlStmt = "insert into " + _dbOwner +".lumi_thresholds(THRESHOLD_ID," 
      " THRESHOLD1_SET1, THRESHOLD1_SET2, THRESHOLD2_SET1,"  
      " THRESHOLD2_SET2, ET_THRESHOLD)VALUES (:tId, :t1S1, :t1S2,"  
      " :t2S1, :t2S2, :te)";
    try {
	_stmt=_conn->createStatement(sqlStmt);
        _stmt->setInt(1, thId);
        _stmt->setInt(2, th1Set1);
        _stmt->setInt(3, th1Set2);
        _stmt->setInt(4, th2Set1);
        _stmt->setInt(5, th2Set2);
        _stmt->setInt(6, thEt);
        _stmt->executeUpdate();
	_conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**
   * Insert Configuration_BITMASKS table. Trigger is not used for BOTMASK_ID because  
   * we will need the id to fill the lumi_section table.
   * June 8, 2007  Y. Guo
   * Aug 22, 2007 J. Jones - changed to pass array pointers rather than by value
   **/
  void DBWriter::insertBind_ConfigBitMask (const long bitMskId,
					  const int mskPlus[18],
					  const int mskMinus[18])
  {
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
      _stmt->setInt (1, bitMskId);
      _stmt->setInt (2, mskPlus[0]);      _stmt->setInt (20, mskMinus[0]);
      _stmt->setInt (3, mskPlus[1]);      _stmt->setInt (21, mskMinus[1]);
      _stmt->setInt (4, mskPlus[2]);      _stmt->setInt (22, mskMinus[2]);
      _stmt->setInt (5, mskPlus[3]);      _stmt->setInt (23, mskMinus[3]);
      _stmt->setInt (6, mskPlus[4]);      _stmt->setInt (24, mskMinus[4]);
      _stmt->setInt (7, mskPlus[5]);      _stmt->setInt (25, mskMinus[5]);
      _stmt->setInt (8, mskPlus[6]);      _stmt->setInt (26, mskMinus[6]);
      _stmt->setInt (9, mskPlus[7]);      _stmt->setInt (27, mskMinus[7]);
      _stmt->setInt (10, mskPlus[8]);     _stmt->setInt (28, mskMinus[8]);
      _stmt->setInt (11, mskPlus[9]);     _stmt->setInt (29, mskMinus[9]);
      _stmt->setInt (12, mskPlus[10]);    _stmt->setInt (30, mskMinus[10]);
      _stmt->setInt (13, mskPlus[11]);    _stmt->setInt (31, mskMinus[11]);
      _stmt->setInt (14, mskPlus[12]);    _stmt->setInt (32, mskMinus[12]);
      _stmt->setInt (15, mskPlus[13]);    _stmt->setInt (33, mskMinus[13]);
      _stmt->setInt (16, mskPlus[14]);    _stmt->setInt (34, mskMinus[14]);
      _stmt->setInt (17, mskPlus[15]);    _stmt->setInt (35, mskMinus[15]);
      _stmt->setInt (18, mskPlus[16]);    _stmt->setInt (36, mskMinus[16]);
      _stmt->setInt (19, mskPlus[17]);    _stmt->setInt (37, mskMinus[17]);
      _stmt->executeUpdate();   
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  } 
  
  /**
   * Function for  bulk bind inseration to all the OCC histogram tables 
   * 
   * Author Y. Guo June 21, 2006
   *
   */
  
  void DBWriter::insertArray_allOccHist(long lsId,
				       int * aBX,
				       unsigned long * aOcc,
				       unsigned long * aLostLnb,
				       int * aHlxNum,
				       int * aRingSetNum,
				       unsigned long aLen,
				       const string & tableName) {
    int Occ_len=0;
    int ntriger = 0;
    int NSEQ = 1000;
    int nseqFetch = 0;
    vector<long> seqSeed;
    
    if(aLen > abs(NSEQ)) {
      ldiv_t q = div(aLen, (long)NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = (int)aLen;
    }
    
    // total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i=0; i<nseqFetch; i++) {
      // get all the sequence seeds we need. The sequence is incremented by 4000
      seqSeed.push_back(getLumiSequence("lumi_all_histgram_seq"));
    }
    
    // construct the sql
    string sqlStmt = "insert into "+ _dbOwner +"." + tableName +
      "(RECORD_ID, SECTION_ID, BUNCH_X_NUMBER,TOTAL_OCCUPANCIES,"
      "HLX_NUMBER, RING_SET_NUMBER, LOST_LNB_COUNT)"
      "VALUES (:rId, :lsId, :bxNum, :occ ,:hlxNum, :rSetNum, :lstLnbCnt)";

    // Fill Section ID
    ub2 aSecLen[aLen];
    long aSec[aLen];

    for(unsigned long i=0; i<aLen; i++) {
      aSec[i]=lsId;   //all data in this set has the same section_id
      aSecLen[i]=sizeof(lsId);
    }
    
    // Fill record_id/sequence_id
    ub2 aSeqLen[aLen];
    long aSeq[aLen];
    int extra=NSEQ;
    for(vector<long>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }    
    
    // Fill for data
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
        _stmt->setDataBuffer(1,(void*)aSeq, OCCIINT, sizeof(aSeq[0]),
			     (unsigned short *)aSeqLen);
        _stmt->setDataBuffer(2,(void*)aSec, OCCIINT, sizeof(aSec[0]),
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
	_stmt->executeArrayUpdate(aLen);
	_conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /***********************************************************
   * 
   * Insert into ET_SUM table
   * Y. Guo    June 8, 2007
   *
   **********************************************************/
  void DBWriter::insertArray_EtSum(long secid,
				  int *aBX,
				  double *aEt, 
				  int *aHlxNum,
				  long *aLostLnbCt,
				  unsigned long aLen) {
    int Occ_len=0;
    int ntriger = 0;
    int NSEQ = 1000;
    int nseqFetch = 0;
    vector<long> seqSeed;

    if (aLen >abs(NSEQ)) {
      ldiv_t q = div(aLen, (long)NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = (int)aLen;
    }

    // total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i = 0; i < nseqFetch; i++) {
      // get all the sequence seeds we need
      long myrecord = getLumiSequence("lumi_all_histgram_seq");
      seqSeed.push_back(myrecord);
    }

    // construct the sql
    string sqlStmt = "insert into "+ _dbOwner+ ".ET_SUMS" 
                     "(record_id, section_id,bunch_x_number,"
                     " ET, HLX_NUMBER, LOST_LNB_COUNT)"
                     " VALUES (:rId, :lsId, :bxnum, :et, :hlxNum, :lstLnbCt)";

    // Fill Section ID
    ub2 aSecLen[aLen];
    long aSec[aLen];
    for(unsigned long i=0; i < aLen; i++) {
        aSec[i]=secid;   // all data in this set has the same section_id
        aSecLen[i]=sizeof(secid);
    }

    // Fill record_id/sequence_id
    ub2 aSeqLen[aLen];
    int aSeq[aLen];
    int extra=NSEQ;
    for(vector<int>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }
    
    // Fill for data
    ub2 aBxLen[aLen], aEtLen[aLen], aLostLnbCtLen[aLen], aHlxNumLen[aLen];
    for(unsigned long i=0; i<aLen; i++){
      aBxLen[i]=sizeof(aBX[i]);
      aEtLen[i]=sizeof(aEt[i]);
      aLostLnbCtLen[i]=sizeof(aLostLnbCt[i]);
      aHlxNumLen[i]=sizeof(aHlxNum[i]);
    }
        
    try{
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,(void*)aSeq, OCCIINT, sizeof(aSeq[0]),(unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,(void*)aSec, OCCIINT, sizeof(aSec[0]), (unsigned short *)aSecLen);
      _stmt->setDataBuffer(3,(void*)(aBX),  OCCIINT, sizeof(aBX[0]),  (unsigned short *)aBxLen);
      _stmt->setDataBuffer(4,(void*)(aEt), OCCIFLOAT, sizeof(aEt[0]), (unsigned short *)aEtLen);
      _stmt->setDataBuffer(5,(void*)(aHlxNum), OCCIINT, sizeof(aHlxNum[0]), 
			   (unsigned short *)aHlxNumLen);
      _stmt->setDataBuffer(6,(void*)(aLostLnbCt), OCCIINT, sizeof(aLostLnbCt[0]), 
			   (unsigned short *)aLostLnbCtLen);      
      _stmt->executeArrayUpdate(aLen);
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /*********************************************************************
   * New signature for lumiSection table inseration. In the new db schema
   * threshod and bit mask are no longer required.
   *
   * Y Guo July 19, 2007
   *
   **********************************************************************/
  void DBWriter::insertBind_LumiSec(long lsId,
				   int setVrsNum,
				   int dataTaking,
				   int beginObt,
				   int totalObt,
				   int runNum,
				   int lsNum,
				   int fillNum,
				   int lsStartT,
				   int lsStopT,
				   const string& comment) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
      "SET_VERSION_NUMBER, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER,"
      "FILL_NUMBER, SEC_START_TIME, SEC_STOP_TIME, COMMENTS)"
      "VALUES (:sid, :setVnum, :dataTk, :bO, :eO,:run, :lsNum,"
      ":fNum,:secStratT,:secStopT,:comm )";
    
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setInt (1, lsId);
      _stmt->setInt (2, setVrsNum);
      _stmt->setInt (3, dataTaking);
      _stmt->setInt (4, beginObt);
      _stmt->setInt (5, totalObt);
      _stmt->setInt (6, runNum);
      _stmt->setInt (7, lsNum);
      _stmt->setInt (8, fillNum);
      _stmt->setInt (9, lsStartT);
      _stmt->setInt (10, lsStopT);
      _stmt->setString (11, comment);
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
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
  void DBWriter::insertBind_LumiSec(long lsId,
				    const DBWriter::DBLumiSection & sectionData) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID,"
      "IS_DATA_TAKING, BEGIN_ORBIT_NUMBER,"
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER)"
      "VALUES (:sid, :dataTk, :bO, :eO,:run, :lsNum)";

    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setInt (1, lsId);
      _stmt->setInt (2, sectionData.dataTaking);
      _stmt->setInt (3, sectionData.beginObt);
      _stmt->setInt (4, sectionData.totalObts);
      _stmt->setInt (5, sectionData.runNum);
      _stmt->setInt (6, sectionData.lsNum);        
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /**********************************************************
   * Another signature for lumiSection table inseration
   * 
   * Y Guo April 18, 2007
   *
   **********************************************************/ 
  void DBWriter::insertBind_LumiSec(long lsId,
				   long thId,
				   long bitmId, 
				   int setVrsNum,
				   int dataTaking,
				   int beginObt, 
				   int totalObt,
				   int runNum,
				   int lsNum,
				   int fillNum,
				   int lsStartT,
				   int lsStopT,
				   const string& comment) {
    string sqlStmt = "insert into " + _dbOwner + ".lumi_sections(SECTION_ID, BITMASK_ID," 
      "THRESHOLD_ID,SET_VERSION_NUMBER, IS_DATA_TAKING, BEGIN_ORBIT_NUMBER," 
      "NUMBER_ORBITS, RUN_NUMBER, LUMI_SECTION_NUMBER," 
      "FILL_NUMBER, SEC_START_TIME, SEC_STOP_TIME, COMMENTS)"
      "VALUES (:sid, :bit ,:th,:setVnum, :dataTk, :bO, :eO,:run, :lsNum," 
      ":fNum,:secStratT,:secStopT,:comm )";
    
    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setInt (1, lsId);
      _stmt->setInt (2, thId);
      _stmt->setInt (3, bitmId);
      _stmt->setInt (4, setVrsNum);
      _stmt->setInt (5, dataTaking);
      _stmt->setInt (6, beginObt);
      _stmt->setInt (7, totalObt);
      _stmt->setInt (8, runNum);
      _stmt->setInt (9, lsNum);
      _stmt->setInt (10, fillNum);
      _stmt->setInt (11, lsStartT);
      _stmt->setInt (12, lsStopT);
      _stmt->setString (13, comment);      
      _stmt->executeUpdate ();
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }
  
  /************************************************************************ 
   * Insert into LUMI_BUNCH_CROSSING Table
   *
   * June 11, 07  Y. Guo
   *
   **********************************************************************/
  void DBWriter::insertArray_LumiBX(long lsId,
				    const DBWriter::DBLumiBX & bxData) {
    int Occ_len=0;
    int ntriger = 0; 
    int NSEQ = 1000;
    int nseqFetch = 0;
    vector<int> seqSeed;
    
    if(bxData.aLen >NSEQ) {
      div_t q = div(bxData.aLen, NSEQ);
      Occ_len = q.rem; //the remainder
      ntriger = q.quot;
    } else {
      Occ_len = bxData.aLen;
    }
    
    // total number of times to get sequqnce
    nseqFetch = ntriger + 1;
    for(int i = 0 ; i < nseqFetch ; i++) {
      // get all the sequence seeds we need
      seqSeed.push_back(getLumiSequence("LUMI_BNCH_X_SEQ" ));
    }

    // construct the sql
    string sqlStmt = "insert into " + _dbOwner + ".lumi_bunch_crossings(RECORD_ID,"
      "SECTION_ID, BUNCH_X_NUMBER, ET_LUMI, ET_LUMI_ERR, ET_LUMI_QLTY," 
      "OCC_LUMI, OCC_LUMI_ERR, OCC_LUMI_QLTY)"
      "VALUES(:rId, :sId, :bxNum, :etL, :etLE, :etLQ, :occL, :occLE," 
      " :occLQ)";

    // Fill lumi section ID
    ub2 aSecLen[bxData.aLen];
    long aSec[bxData.aLen];
    for(int i = 0; i < bxData.aLen; i++) {
      aSec[i] = lsId;   //all data in this set has the same section_id
      aSecLen[i] = sizeof(lsId);
    }
    
    // Fill record_id/sequence_id
    ub2 aSeqLen[bxData.aLen];
    int aSeq[bxData.aLen];
    int extra=NSEQ;
    for(vector<int>::size_type j=0; j < abs(nseqFetch); j++) {
      if(j==abs(nseqFetch-1))extra=Occ_len;
      for(unsigned int i=j*NSEQ; i<(j*NSEQ+extra); i++) {
	aSeq[i]=seqSeed[j]+(i - j*NSEQ);
	aSeqLen[i]=sizeof(aSeq[i]);
      }
    }

    // Fill for data
    ub2 aBxLen[bxData.aLen], aEtLumiLen[bxData.aLen], aEtErrLen[bxData.aLen], aEtQLen[bxData.aLen];
    ub2 aOccLumiLen[bxData.aLen], aOccErrLen[bxData.aLen], aOccQLen[bxData.aLen];
    for(int i=0; i < bxData.aLen; i++) {
      aBxLen[i]=sizeof(bxData.aBX[i]);
      aEtLumiLen[i]=sizeof(bxData.aEtLumi[i]);
      aEtErrLen[i]=sizeof(bxData.aEtErr[i]);
      aEtQLen[i]=sizeof(bxData.aEtQ[i]);
      aOccLumiLen[i]=sizeof(bxData.aOccLumi[i]);
      aOccErrLen[i]=sizeof(bxData.aOccErr[i]);
      aOccQLen[i]=sizeof(bxData.aOccQ[i]);
      /**
	 cout << "i=" <<i <<": "<<aBxLen[i] <<","<<aEtLumiLen[i]<<","<<aEtErrLen[i]<<",";
	 cout << aEtQLen[i] <<","<< aOccLumiLen[i] <<","<< aOccErrLen[i] <<",";
	 cout << aOccQLen[i] <<endl;
	 cout << aBX[i] <<"," << aEtLumi[i] <<"," <<aEtErr[i] <<",";
	 cout << aEtQ[i] <<"," << aOccLumi[i] << "," << aOccErr[i] <<","<<aOccQ[i]<<endl;
      **/ 
    }
    
    try{
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setDataBuffer(1,
			   (void*)aSeq,
			   OCCIINT,
			   sizeof(aSeq[0]),
			   (unsigned short *)aSeqLen);
      _stmt->setDataBuffer(2,
			   (void*)aSec,
			   OCCIINT,
			   sizeof(aSec[0]),
			   (unsigned short *)aSecLen);    
      _stmt->setDataBuffer(3,
			   (void*)bxData.aBX,
			   OCCIINT,
			   sizeof(bxData.aBX[0]),
			   (unsigned short *)aBxLen);
      _stmt->setDataBuffer(4,
			   (void*)bxData.aEtLumi,
			   OCCIFLOAT,
			   sizeof(bxData.aEtLumi[0]), 
			   (unsigned short *)aEtLumiLen);
      _stmt->setDataBuffer(5,
			   (void*)bxData.aEtErr,
			   OCCIFLOAT,
			   sizeof(bxData.aEtErr[0]), 
			   (unsigned short *)aEtErrLen);
      _stmt->setDataBuffer(6,
			   (void*)bxData.aEtQ,
			   OCCIINT,
			   sizeof(bxData.aEtQ[0]), 
			   (unsigned short *)aEtQLen);
      _stmt->setDataBuffer(7,
			   (void*)bxData.aOccLumi,
			   OCCIFLOAT,
			   sizeof(bxData.aOccLumi[0]),
			   (unsigned short *)aOccLumiLen);
      _stmt->setDataBuffer(8,
			   (void*)bxData.aOccErr,
			   OCCIFLOAT,
			   sizeof(bxData.aOccErr[0]),
			   (unsigned short *)aOccErrLen);
      _stmt->setDataBuffer(9,
			   (void*)bxData.aOccQ,
			   OCCIINT,
			   sizeof(bxData.aOccQ[0]),
			   (unsigned short *)aOccQLen);
      _stmt->executeArrayUpdate(bxData.aLen);
      _conn->terminateStatement (_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

  /**************************************************************************
   *
   * Insert into Lumi_Summary Table.
   *
   * June 11, 07   Y. Guo
   *
   **************************************************************************/
  void DBWriter::insertBind_LumiSummary(long lsId,
					const DBWriter::DBLumiSummary & summary) {
    string sqlStmt = "insert into "+ _dbOwner + ".LUMI_SUMMARIES(SECTION_ID,"
      "DEADTIME_NORMALIZATION,"
      "NORMALIZATION, INSTANT_LUMI, INSTANT_LUMI_ERR," 
      "INSTANT_LUMI_QLTY, INSTANT_ET_LUMI, INSTANT_ET_LUMI_ERR," 
      "INSTANT_ET_LUMI_QLTY, INSTANT_OCC_LUMI, INSTANT_OCC_LUMI_ERR," 
      "INSTANT_OCC_LUMI_QLTY) VALUES(:sId, :deadT, :nor, :dLumi,"
      ":dLumiErr, :dLumiQ, :dEtLumi, :dEtErr, :dEtQ, :dOccLumi, :dOccErr,"
      ":dOccQ)";

    try {
      _stmt=_conn->createStatement (sqlStmt);
      _stmt->setInt(1, lsId);
      _stmt->setDouble(2, summary.dTimeNorm);
      _stmt->setDouble(3, summary.norm);
      _stmt->setDouble(4, summary.instLumi);
      _stmt->setDouble(5, summary.instLumiErr);
      _stmt->setInt(6, summary.instLumiQ);
      _stmt->setDouble(7, summary.instEtLumi);
      _stmt->setDouble(8, summary.instEtLumiErr);
      _stmt->setInt(9, summary.instEtLumiQ);
      _stmt->setDouble(10, summary.instOccLumi);
      _stmt->setDouble(11, summary.instOccLumiErr);
      _stmt->setInt(12, summary.instOccLumiQ);
      _stmt->executeUpdate();
      _conn->terminateStatement(_stmt);
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);      
    }
  }
  
  void DBWriter::DisplayDisTableBX(int secid){
    string  sqlQuery = "SELECT  BUNCH_X_NUMBER, DISABLE_OCCUPANCIES, "
      "LOST_PKT_NUMBER FROM CMS_LUMI_HF_OWNER.DISABLE_OCCUPANCIES " 
      "where SECTION_ID = :y order by BUNCH_X_NUMBER"; 
    
    try {
      _stmt = _conn->createStatement (sqlQuery);    
      _stmt->setInt (1, secid);   
      ResultSet *rset = _stmt->executeQuery ();

      // cout what we're doing
      cout << "Displaying Disable Table" << endl;
      cout << "sectionId"
	   << setw(10)
	   << "BX"
	   << setw(15)
	   << "disOccupancy"
	   << setw(10)
	   << "lostpkts"
	   << endl;
      
      while (rset->next()){
	cout << (int)rset->getInt(1)
	     << setw(5)
	     << (int)rset->getInt(2)
	     << setw(5)
	     << (int)rset->getInt(3)
	     << endl;
      }

      _stmt->closeResultSet (rset);
      _conn->terminateStatement(_stmt);      
    } catch (SQLException aExc) {
      OracleDBException lExc(aExc.getMessage());
      RAISE(lExc);
    }
  }

}
