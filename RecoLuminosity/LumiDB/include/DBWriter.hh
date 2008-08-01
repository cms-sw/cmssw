#ifndef LUMIDB_lumiDbServices_cppAccess_DBWrite_h
#define LUMIDB_lumiDbServices_cppAccess_DBWrite_h

/******************************************************************
 *  This API is used online for lumi server to write to lumi db.
 *  
 *  Author: Yuyi Guo                                          
 *  Updates made by John Jones to improve exception handling
 *  and reduce stack usage in function calls
 *  Creation Date: June 07
 *  $Id: DBWriter.hh,v 1.3 2008/01/09 14:02:08 jaj99 Exp $
 ******************************************************************/

#include <iostream>

#include "occi.h"

#include <map>
#include <string.h>

namespace HCAL_HLX {
  
  class DBWriter {
  public:
    
    DBWriter(const std::string & user,
	     const std::string & passwd,
	     const std::string & db,
	     const std::string & dbOwner);
    ~DBWriter ();
    void save();
    
    oracle::occi::Connection* getConnection() const { return _conn;}
    //Get the Max run number in the cms_runs table. 
    unsigned int maxRunNumber( );
    
    //CONFIGURATION_BITMASKS and lumi_thresholds table are sharing the same seq number.
    unsigned long getThresholdConfigMId();
    
    //get the general purpose sequence.
    unsigned long getLumiSequence( const std::string& seqName);
    
    //get the sequence for lumi_tags table
    unsigned long getLumiTagSeq();
    
    //get the sequence for lumi_sections table.
    unsigned long getLumiSectionSeq();
    
    //get the sequence for TRG_AND_DEADTIME_DEFINITIONS table.
    unsigned long getTrgDefSeq();
    
    //get the sequence for LEVEL1_TRIGGERS.
    unsigned long getL1TrgSeq();
    
    //get sequence for TRIGGER_DEADTIMES.
    unsigned long getTrgDtimeSeq();
    
    //get sequence from cRun_seq
    unsigned long getCRunSeq();
    
    //insert into cms_runs table
    void insertCmsRuns(unsigned int runNum);
    
    //query LUMI_HF_RING_SET table
    unsigned long getHFRingSetId(unsigned int setVrsnNum);
    
    //get the deadtime definition from TRIGGER_DEADTIME_DEF table.
    std::map<std::string, unsigned long> & getTrgDTDefMap();
    
    /*Fill _triggerDefinitionMap or _deadtimeDefinitionMap.
     *
     */
    void putTrgDeadtimeDefMap();
    
    /*Insert a set of deadtime into TRIGGER_DEADTIMES table for the same lumi section.
     *lsId: lumi section id; 
     *aDeadtimeDef: an array of deadtime defination from the definition table.
     *aDeadtime: an array of deadtime. The value is 0-2^32. 
     *aLen: the length of the arrays. 
     */ 
    void insertArray_deadtime(unsigned long lsId,
			      unsigned long* aDeadtimeDefId,
			      unsigned int* aDeadtime,
			      unsigned int aLen);
    
    /*Insert a set of L1 triggers into LEVEL1_TRIGGER table 
     *for the same lumi section.
     *lsId: lumi section id; 
     *aL1LnNum: an array of trigger line numbers that from 1 to 128. 
     *aL1Scl: an array of L1_scaler values.
     *aL1RtCnt: an array of L1_scaler values.
     *aHLTInt: an array of HLT_INPUT values. 
     *aLen: the length of the arrays
     */
    void insertArray_L1_trigger(unsigned long lsId, 
				unsigned int* aL1LnNum,
				unsigned int* aL1Scl,
				unsigned int* aL1RtCnt,
				unsigned int* aHLTInt,
				unsigned int aLen);
    
    /*Insert a set of HLT trigger into HLTS table for the same lumi section.
     *lsId: lumi section id;
     *aTrgPth: an array of trigger_path values.
     *aInptCnt: an array of INPUT_COUNT values. 
     *aAccptCnt: an array of ACCEPT_COUNT values.
     *aPreSclFct: an array of PRESCALE_FACTOR values.
     *aLen: the length of the arrays
     */
    void insertArray_HLTs(unsigned long lsId, const char* const* aTrgPth,
			  unsigned long* aInptCnt, unsigned long* aAccptCnt,
			  unsigned int* aPreSclFct, unsigned int aLen);
    
    /* Insert into HLT 
     *
     */
    void insertArray_HLTs(unsigned long lsId, char* aTrgPth,
			  unsigned long* aInptCnt, unsigned long* aAccptCnt,
			  unsigned int* aPreSclFct, unsigned int aLen);  
    
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
    void insertBind_LumiSec(unsigned long lsId,
			    unsigned long thId,
			    unsigned long bitmId, 
			    int setVrsNum,
			    int dataTaking,
			    unsigned int beginObt,
			    unsigned int totalObt,
			    unsigned int runNum, 
			    unsigned int lsNum,
			    unsigned int fillNum,
			    unsigned long lsStartT,
			    unsigned long lsStopT, 
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
    void insertBind_LumiSec(unsigned long lsId,
			    int HFRngStId,
			    int dataTaking,
			    unsigned int beginObt,
			    unsigned int totalObt,  
			    unsigned int runNum,
			    unsigned int lsNum,
			    unsigned int fillNum,
			    unsigned long lsStartT,
			    unsigned long lsStopT, 
			    const std::string& comment);
    
    /*Insert into lumi_sections table with bare minimum data.
     *lsId: lumi section ID from orcale sequence generator.
     *dataTaking: data taking flag. 1 for taking physics data, 0 for everything else.
     *beginObt: the begining orbit number for the lumi section.
     *totalObt: total number of orbits in the lumi section.
     *runNum: run number that the lumi section belongs to.
     *lsNum: lumi section number that from trigger or LHC.
     */ 
    //    void insertBind_LumiSec(unsigned long lsId,
    //			    unsigned int dataTaking, unsigned int beginObt,
    //			    unsigned int totalObts,  unsigned int runNum,  unsigned int lsNum, 
    //			    unsigned long HFRngStId);
    
    //void insertBind_LumiSec(unsigned long lsId, unsigned long HFRngStId,
    //unsigned int dataTaking, unsigned int beginObt,
    //unsigned int totalObts,  unsigned int runNum,  unsigned int lsNum);
    
    struct DBLumiSection {
      unsigned long HFringStId; // and this is???
      unsigned int dataTaking;
      unsigned int beginObt;
      unsigned int totalObts;
      unsigned int runNum;
      unsigned int lsNum;
    };
    void insertBind_LumiSec(unsigned long lsId,
			    const DBWriter::DBLumiSection & sectionData);
    
    /*Insert into LUMI_DETAILS table.
     *lsId: lumi_section_id as foregin key from lumi_sections table.
     *aBx: an array of bunch number in the lumi section.
     *aEtLumi: an array of luminosity calculated by et sum for each bunch corssing number in aBx.
     *aEtErr: an array of static error for aEtLumi.
     *aEtQ: an array of quality flag for aEtLum.
     *aOccLumi: an array of luminosity calculated by occpuanies for each bunch corssing number 
     *in aBx.
     *aOccErr: an array of static error for aOccLumi.
     *aOccQ: an array of quality flags for aOccLumi.
     *alen: the length of the arrays.  
     */

    struct DBLumiDetails {
      unsigned int* aBX;
      float* aNorEtLumi;
      float* aEtLumi;
      float* aEtLumiErr;
      unsigned int* aEtLumiQ;
      float* aNorOccLumiD1;
      float* aOccLumiLumiD1;
      float* aOccLumiD1Err;
      unsigned int* aOccLumiD1Q;
      float* aNorOccLumiD2;
      float* aOccLumiLumiD2;
      float* aOccLumiD2Err;
      unsigned int* aOccLumiD2Q;
      unsigned int aLen;
    };
    void insertArray_LumiDetails(unsigned long lsId,
				 const DBWriter::DBLumiDetails & details);
    
    
    /*Insert into LUMI_SUMMARIES table.
     *lsId: lumi_section_id as foregin key from lumi_sections table.
     *dTimeNorm: normalized deadtime for this lumi section.
     *norm: normalization for INSTANT_LUMI
     *instLumi: instance luminosity for this lumi section.
     *instLumiErr: static error for instLumi.
     *instLumiQ: quality flag for instLumi.
     *normEt: normalization for INSTANT_ET_LUMI.
     *instEtLumi: instance luminosity calcuated by et sum.
     *instEtLumiErr: static error for instEtLumi.
     *instEtLumiQ: quality flag for instEtLumi.
     *norOcc1: normalization for INSTANT_OCC_LUMI_D1
     *instOccLumiD1: instance luminosity calcuated by occpuancy.
     *instOccLumiD1Err: static error for instOccLumiD1.
     *instOccLumiD1Q: quality flag for instOccLumiD1.
     *norOcc2: normalization for INSTANT_OCC_LUMI_D2
     *instOccLumiD2: instance luminosity calcuated by occpuancy.
     *instOccLumiD2Err: static error for instOccLumiD2.
     *instOccLumiD2Q: quality flag for instOccLumiD2.

     */

  struct DBLumiSummary {
    float dTimeNorm;
    float norm;
    float instLumi;
    float instLumiErr;
    unsigned int instLumiQ;
    float norEt;
    float instEtLumi;
    float instEtLumiErr;
    unsigned int instEtLumiQ;
    float norOccLumiD1;
    float instOccLumiD1;
    float instOccLumiD1Err;
    unsigned int instOccLumiD1Q;
    float norOccLumiD2;
    float instOccLumiD2;
    float instOccLumiD2Err;
    unsigned int instOccLumiD2Q;
  };
  void insertBind_LumiSummary(unsigned long lsId,
			      const DBWriter::DBLumiSummary & summary);
  
  /*Get lumi_summary_id from lumi_summarys table with given lumi_version and section_id.
   *lsId: section_id 
   *lumiVer: lumi_version
   */
  unsigned long getLumiSummaryId(unsigned long lsId, int lumiVer);
  
  /* Create a new lumi Tag into lumi_tags table.
   * tName: tag name to be created.
   */
  unsigned long insertLumiTag(std::string& tName);
  
  /* Get lumi_tag_id of TAG_NAME tName from lumi_tags table.
   * tName: tag name in the table.
   */
  unsigned long getLumiTagId(std::string& tName);
  
  /* Insert into lumi_version_tag_maps table. The API will find the 
   * lumi_summary_id assocated with lumi_version, lumi_section_number and run_number,
   * as well as lumi_tag_id of tag_name. Then insert lumi_summary_id and lumi_tag_id 
   * pair into the maps.
   * lsNum: lumi_section_number.
   * runNum: run number.
   * lsVer: lumi version numner.
   * tName: Tag name.
   */
  void insertLumiVerTagMap(unsigned int lsNum, unsigned int runNum, int lsVer,                                 std::string& tName);
  
  /*Get lumi section id for a pair of lumi_section_number and run_number.
   *lsNum: lumi section number.
   *runNum: run number.
   */
  unsigned long getLumiSecId(unsigned int runNum, unsigned int lsNum);
  
  /*Not used yet. Maybe need it in the future.
   */
  void insertBind_LumiTrigger(unsigned int  runNum, unsigned int lsNum, 
			      const std::string& triggerType, 
			      const std::string& triggerPath, unsigned int inputCount, 
			      unsigned int accptCount, 
			      const std::string& prescaleModule, unsigned int prescaleFactor);
  /*
   */
  void insertBind_threshold (unsigned long thId, int th1Set1, int th1Set2, int th2Set1,
			     int th2Set2, int thEt);
  /*
   */
  void insertBind_ConfigBitMask(const long bitMskId,
				const int mskForward[18],
				const int mskBackward[18]);
  
  /*
   */
  void insertArray_allOccHist(unsigned long lsId,
			      int* aBX,
			      unsigned long* aOcc,
			      unsigned long* aLostLnb,
			      int* aHlxNum,
			      int* aRingSetNum,
			      unsigned long aLen,
			      const std::string & tableName);

  /*
   */
  void insertArray_EtSum(unsigned long secid,
			 int* aBX,
			 double* aEt,
			 int* aHlxNum,
			 long* aLostLnbCt,
			 unsigned long aLen);
  
private:
  oracle::occi::Environment *_env;
  oracle::occi::Connection *_conn;
  oracle::occi::Statement *_stmt;

  // DB schema owner that is different from the writer. 
  std::string _dbOwner;

  // Map for trigger definition for quick access.
  //std::map<std::string, int>_triggerDefinitionMap;

  // Map for deadtime definition for quick access
  std::map<std::string, unsigned long>_deadtimeDefinitionMap; 
};

}

#endif // DataFormats_Luminosity_LumiSummary_h
