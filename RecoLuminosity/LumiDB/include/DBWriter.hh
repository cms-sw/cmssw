#ifndef LUMIDB_lumiDbServices_cppAccess_DBWrite_h
#define LUMIDB_lumiDbServices_cppAccess_DBWrite_h

/******************************************************************
 *  This API is used online for lumi server to write to lumi db.
 *  
 *  Author: Yuyi Guo                                          
 *  Updates made by John Jones to improve exception handling
 *  and reduce stack usage in function calls
 *  Creation Date: June 07
 *  $Id: DBWrite.hh,v 1.5 2007/08/21 22:02:19 yuyi Exp $
 ******************************************************************/

#include <iostream>
#include <occi.h>
#include <map>
#include <string.h>

class DBWriter {
public:
  
  DBWriter(const std::string & user,
	  const std::string & passwd,
	  const std::string & db,
	  const std::string & dbOwner);
  ~DBWriter ();
  
  oracle::occi::Connection* getConnection() const { return _conn;}
  //Get the Max run number in the lumi_sections table. 
  int maxRunNumber( );
  
  //CONFIGURATION_BITMASKS and lumi_thresholds table are sharing the same seq number.
  long getThresholdConfigMId();
  
  //get the general purpose sequence.
  long getLumiSequence( const std::string& seqName);
  
  //get the sequence for lumi_sections table.
  long getLumiSectionSeq();
  
  //get the sequence for TRG_AND_DEADTIME_DEFINITIONS table.
  long getTrgDefSeq();
  
  //get the sequence for LEVEL1_HLT_TRIGGERS.
  long getTrgSeq();
  
  //get sequence for TRIGGER_DEADTIMES.
  long getTrgDtimeSeq();
  
  //get the trigger definition from TRG_AND_DEADTIME_DEFINITIONS table.
  const std::map<std::string, int> & getTriggerDefinitionMap();
  
  //get the deadtime definition from TRG_AND_DEADTIME_DEFINITIONS table.
  const std::map<std::string, int> & getDeadtimeDefinitionMap();
  
  //insert trigger definition into TRG_AND_DEADTIME_DEFINITIONS table.
  void putTriggerDefinitionMap();
  
  //insert deadtime definition into TRG_AND_DEADTIME_DEFINITIONS table.
  void putDeadtimeDefinitionMap();
  
  /*Fill _triggerDefinitionMap or _deadtimeDefinitionMap. input=1 for deadtime
   *input=0 for trigger.
   */
  //void putTriggerDeadtimeDefinitionMap(int);
  
  /*Insert a set of deadtime into TRIGGER_DEADTIMES table for the same lumi section.
   *lsId: lumi section id; 
   *aDeadtimeDef: an array of deadtime defination from the definition table.
   *aDeadtime: an array of deadtime. Note Valerie moention that these are big integers, 
   *but not sure how big they are. The max long is 2143483647(2^31-1) on 32 bit machines. 
   *aLen: the length of the arrays. 
   */ 
  void insertArray_deadtime(long lsId,
			    int *aDeadtimeDef,
			    long *aDeadtime,
			    unsigned int aLen);
  
  /*Insert a set of level1 triggers into LEVEL1_HLT_TRIGGERS table for the same lumi section.
   *lsId: lumi section id; 
   *aTrgDefId: an array of trigger defination from the definition table.
   *aTrgBitNum: an array of trigger bit numbers that from 1 to 128. 
   *aTrgVal: an array of trigger values.
   *aLen: the length of the arrays
   */
  void insertArray_trigger(long lsId,
			   int *aTrgDefId,
			   int *aTrgBitNum,
			   int *aTrgVal,
			   unsigned int aLen);
  
  /*Insert a set of HLT trigger into LEVEL1_HLT_TRIGGERS table for the same lumi section.
   *lsId:lumi section id; 
   *aTrgDefId: an array of trigger defination from the definition table.
   *aTrgVal: an array of trigger values.
   *aLen: the length of the arrays
   */
  void insertArray_trigger(long lsId,
			   int *aTrgDefId,
			   int *aTrgVal,
			   unsigned int aLen);
  
  /*
   */
  void insertBind_threshold(long thId,
			    int th1Set1,
			    int th1Set2,
			    int th2Set1,
			    int th2Set2,
			    int thEt);
  /*
   */
  void insertBind_ConfigBitMask(const long bitMskId,
				const int mskPlus[18],
				const int mskMinus[18]);
  
  /*
   */
  void insertArray_allOccHist(long lsId,
			      int *aBX,
			      unsigned long *aOcc,
			      unsigned long *aLostLnb,
			      int *aHlxNum,
			      int *aRingSetNum,
			      unsigned long aLen,
			      const std::string & tableName);
  /*
   */
  void insertArray_EtSum(long secid,
			 int *aBX,
			 double *aEt,
			 int *aHlxNum,
			 long *aLostLnbCt,
			 unsigned long aLen);
  
  /*Insert into lumi_sections table.
   *lsId: lumi section ID from orcale sequence generator.
   *thId: threshold_ID from LUMI_THRESHOLDS table as a foreign key.
   *bitmId: bitmask_id from CONFIGURATION_BITMASKS table as foreign key.
   *setVrsNum: set_version_number from LUMI_HF_RING_SETS table. It will not change untill 
   *hardware configuration changes.
   *dataTaking: data taking flag. 1 for taking physics data, 0 for everything else.
   *beginObt: the begining orbit number for the lumi section.
   *totalObt: total number of orbits in the lumi section.
   *runNum: run number that the lumi section belongs to.
   *lsNum: lumi section number that from trigger or LHC.
   *fillNum: used to replace run number when no physics runs. It may be dropped in the future.
   *lsStartT: lumi section start time.
   *lsStopT: lumi section stop time.
   *comment: anything you want to say about the current lumi section. 
   */
  void insertBind_LumiSec(long lsId,
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
			  const std::string & comment);
  
  /*Insert into lumi_sections table without foregin keys from threshold and bitmask.
   *lsId: lumi section ID from orcale sequence generator.
   *setVrsNum: set_version_number from LUMI_HF_RING_SETS table. It will not change untill 
   *hardware configuration changes.
   *dataTaking: data taking flag. 1 for taking physics data, 0 for everything else.
   *beginObt: the begining orbit number for the lumi section.
   *totalObt: total number of orbits in the lumi section.
   *runNum: run number that the lumi section belongs to.
   *lsNum: lumi section number that from trigger or LHC.
   *fillNum: used to replace run number when no physics runs. It may be dropped in the future.
   *lsStartT: lumi section start time.
   *lsStopT: lumi section stop time.
   *comment: anything you want to say about the current lumi section.
   */
  void insertBind_LumiSec(long lsId,
			  int setVrsNum,
			  int dataTaking,
                          int beginObt,
			  int endObt,
			  int runNum,
			  int lsNum,
			  int fillNum,
                          int lsStartT,
			  int lsStopT,
			  const std::string & comment);
  
  /*Insert into lumi_sections table with bare minimum data.
   *lsId: lumi section ID from orcale sequence generator.
   *dataTaking: data taking flag. 1 for taking physics data, 0 for everything else.
   *beginObt: the begining orbit number for the lumi section.
   *totalObt: total number of orbits in the lumi section.
   *runNum: run number that the lumi section belongs to.
   *lsNum: lumi section number that from trigger or LHC.
   */ 

  struct DBLumiSection {
    int dataTaking;
    int beginObt;
    int totalObts;
    int runNum;
    int lsNum;
  };
  void insertBind_LumiSec(long lsId,
			  const DBWriter::DBLumiSection & sectionData);
  
  /*Insert into LUMI_BUNCH_CROSSINGS table.
   *lsId: lumi_section_id as foregin key from lumi_sections table.
   *aBx: an array of bunch number in the lumi section.
   *aEtLumi: an array of luminosity calculated by et sum for each bunch crossing number in aBx.
   *aEtErr: an array of static error for aEtLumi.
   *aEtQ: an array of quality flag for aEtLum.
   *aOccLumi: an array of luminosity calculated by occpuanies for each bunch crossing number 
   *in aBx.
   *aOccErr: an array of static error for aOccLumi.
   *aOccQ: an array of quality flags for aOccLumi.
   *alen: the length of the arrays.  
   */
  struct DBLumiBX {
    int aLen;
    int *aBX;
    double *aEtLumi;
    double *aEtErr;
    int *aEtQ;
    double *aOccLumi;
    double *aOccErr;
    int *aOccQ;
  };
  void insertArray_LumiBX(long lsId,
			  const DBWriter::DBLumiBX & bxData);
  
  
  /*Insert into LUMI_SUMMARIES table.
   *lsId: lumi_section_id as foregin key from lumi_sections table.
   *dTimeNorm: normalized deadtime for this lumi section.
   *norm: second level normalization. if none, set it to 1.
   *instLumi: instance luminosity for this lumi section.
   *instLumiErr: static error for instLumi.
   *instLumiQ: quality flag for instLumi.
   *instEtLumi: instance luminosity calcuated by et sum.
   *instEtLumiErr: static error for instEtLumi.
   *instEtLumiQ: quality flag for instEtLumi.
   *instOccLumi: instance luminosity calcuated by occpuancy.
   *instOccLumiErr: static error for instOccLumi.
   *instOccLumiQ: quality flag for instOccLumi.
   */
  struct DBLumiSummary {
    double dTimeNorm;
    double norm;
    double instLumi;
    double instLumiErr;
    int instLumiQ;
    double instEtLumi;
    double instEtLumiErr;
    int instEtLumiQ;
    double instOccLumi;
    double instOccLumiErr;
    int instOccLumiQ;
  };
  void insertBind_LumiSummary(long lsId,
			      const DBWriter::DBLumiSummary & summary);
  
  /*Not used yet.
   */
  void DisplayDisTable(int secid);
  
  /*Not used yet.
   */
  void DisplayDisTableBX(int secid);
  
  
  /*Not used yet. Maybe need it in the future.
   */
  void insertBind_LumiTrigger(long runNum,
			      int lsNum,
			      const std::string & triggerType, 
			      const std::string & triggerPath,
			      long inputCount,
			      long accptCount, 
			      const std::string & prescaleModule,
			      long prescaleFactor);
  
private:
  oracle::occi::Environment *_env;
  oracle::occi::Connection *_conn;
  oracle::occi::Statement *_stmt;

  // DB schema owner that is different from the writer. 
  std::string _dbOwner;

  // Map for trigger definition for quick access.
  std::map<std::string, int>_triggerDefinitionMap;

  // Map for deadtime definition for quick access
  std::map<std::string, int>_deadtimeDefinitionMap; 
};

#endif // DataFormats_Luminosity_LumiSummary_h
