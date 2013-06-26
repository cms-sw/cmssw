#ifndef RUNDCSHVEBDAT_H
#define RUNDCSHVEBDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/DataReducer.h"
#include "OnlineDB/Oracle/interface/Oracle.h"

class RunDCSHVDat : public IDataItem {
 public:
  typedef oracle::occi::ResultSet ResultSet;

  static const int maxDifference = 30*60*1000000; // 30 minutes
  static const int maxHVDifferenceEB = 300;       // max HV tolerance in mV for EB
  static const int maxHVDifferenceEE = 5000;         // max HV tolerance in mV for EE
  static const int minHV = 10000;                 // if HV less than this value (in mV) HV is off
  
  static const int HVNOTNOMINAL = 1;
  static const int HVOFF        = 2;

  friend class EcalCondDBInterface;
  RunDCSHVDat();
  ~RunDCSHVDat();

  // User data methods
  inline std::string getTable() { return ""; }
  inline std::string getEBAccount() { return "CMS_ECAL_HV_PVSS_COND"; }
  inline std::string getEEAccount() { return "CMS_EE_HV_PVSS_COND"; }
  inline void setHV(float t) { m_hv = t; }
  inline void setStatus(int t) { m_status = t; }
  inline void setHVNominal(float t) { m_hvnom = t; }
  inline float getHV() const { return m_hv; }
  inline float getHVNominal() const { return m_hvnom; }
  inline int getStatus() const { return m_status; }
  int getTimeStatus() {return m_tstatus;}
  void setTimeStatus(int t ) {m_tstatus=t; } 
 private:
  void setStatusForBarrel(RunDCSHVDat&, Tm);
  void setStatusForEndcaps(RunDCSHVDat&, Tm);
  ResultSet* getBarrelRset();
  ResultSet* getBarrelRset(Tm timeStart) ;

  ResultSet* getEndcapAnodeRset();
  ResultSet* getEndcapDynodeRset();

  ResultSet* getEndcapAnodeRset(Tm timestart);
  ResultSet* getEndcapDynodeRset(Tm timestart);
  int nowMicroseconds();
  void fillTheMap(ResultSet *, std::map< EcalLogicID, RunDCSHVDat >* );
  //  void fillTheMapByTime(ResultSet *, std::list< std::pair< Tm, std::map< EcalLogicID, RunDCSHVDat > > >* ) ;
  void fillTheMapByTime(ResultSet *rset, std::list<  DataReducer<RunDCSHVDat>::MyData<RunDCSHVDat>  >* my_data_list ) ;


  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const RunDCSHVDat* item, RunIOV* iov )
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, RunDCSHVDat >* fillMap, RunIOV* iov)
     throw(std::runtime_error);

  void fetchLastData(std::map< EcalLogicID, RunDCSHVDat >* fillMap)
     throw(std::runtime_error);

  void fetchHistoricalData(std::list< std::pair<Tm, std::map< EcalLogicID, RunDCSHVDat > > >* fillMap, Tm timeStart  )
    throw(std::runtime_error);


  // User data
  float m_hv;
  float m_hvnom;
  int m_status;
  int m_tstatus; 
};

#endif
