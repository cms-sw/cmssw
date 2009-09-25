#ifndef ODECALCYCLE_H
#define ODECALCYCLE_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODEcalCycle : public IODConfig {
 public:
  friend class EcalCondDBInterface ;

  ODEcalCycle();
  ~ODEcalCycle();

  // User data methods
  inline std::string getTable() { return "ECAL_CYCLE"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }
  inline void setTag(std::string x) { m_tag = x; }
  inline std::string getTag() const { return m_tag; }
  inline void setVersion(int x) { m_version = x; }
  inline int getVersion() const { return m_version; }
  inline void setSeqNum(int x) { m_seq_num = x; }
  inline int getSeqNum() const { return m_seq_num; }

  inline void setSequenceId(int x) { m_seq_id = x; }
  inline int getSequenceId() const { return m_seq_id; }

  inline void setCycleNum(int x) { m_cycle_num = x; }
  inline int getCycleNum() const { return m_cycle_num; }
  inline void setCycleTag(std::string x) { m_cycle_tag = x; }
  inline std::string getCycleTag() const { return m_cycle_tag; }
  inline void setCycleDescription(std::string x) { m_cycle_description = x; }
  inline std::string getCycleDescription() const { return m_cycle_description; }
  inline void setCCSId(int x) { m_ccs = x; }
  inline int getCCSId() const { return m_ccs; }
  inline void setDCCId(int x) { m_dcc = x; }
  inline int getDCCId() const { return m_dcc; }
  inline void setLaserId(int x) { m_laser = x; }
  inline int getLaserId() const { return m_laser; }
  inline void setLTCId(int x) { m_ltc = x; }
  inline int getLTCId() const { return m_ltc; }
  inline void setLTSId(int x) { m_lts = x; }
  inline int getLTSId() const { return m_lts; }
  inline void setDCUId(int x) { m_dcu = x; }
  inline int getDCUId() const { return m_dcu; }
  inline void setTCCId(int x) { m_tcc = x; }
  inline int getTCCId() const { return m_tcc; }
  inline void setTCCEEId(int x) { m_tcc_ee = x; }
  inline int getTCCEEId() const { return m_tcc_ee; }
  inline void setTTCCIId(int x) { m_ttcci = x; }
  inline int getTTCCIId() const { return m_ttcci; }
  inline void setMataqId(int x) { m_mataq = x; }
  inline int getMataqId() const { return m_mataq; }
  inline void setJBH4Id(int x) { m_jbh4 = x; }
  inline int getJBH4Id() const { return m_jbh4; }
  inline void setScanId(int x) { m_scan = x; }
  inline int getScanId() const { return m_scan; }
  inline void setTTCFId(int x) { m_ttcf = x; }
  inline int getTTCFId() const { return m_ttcf; }
  inline void setSRPId(int x) { m_srp = x; }
  inline int getSRPId() const { return m_srp; }

  void printout();

 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODEcalCycle * result)     throw(std::runtime_error);

  // User data
  int m_ID;
  std::string m_tag;
  int m_version;
  int m_seq_num;
  int m_seq_id;
  int m_cycle_num;
  std::string m_cycle_tag;
  std::string m_cycle_description;
  int m_ccs;
  int m_dcc;
  int m_laser;
  int m_ltc;
  int m_lts;
  int m_dcu;
  int m_tcc;
  int m_tcc_ee;
  int m_ttcci;
  int m_mataq;
  int m_jbh4;
  int m_scan;
  int m_srp;
  int m_ttcf;

};

#endif
