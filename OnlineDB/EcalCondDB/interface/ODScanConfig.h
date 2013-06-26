#ifndef ODSCANCONFIG_H
#define ODSCANCONFIG_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODScanConfig : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODScanConfig();
  ~ODScanConfig();

  // User data methods
  inline std::string getTable() { return "ECAL_Scan_DAT"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }

  inline void setTypeId(int x) { m_type_id = x; }
  inline int getTypeId() const { return m_type_id; }

  inline void setScanType(std::string x) { m_type = x; }
  inline std::string getScanType() const { return m_type; }

  inline void setFromVal(int x) { m_from_val = x; }
  inline int getFromVal() const { return m_from_val; }

  inline void setToVal(int x) { m_to_val = x; }
  inline int getToVal() const { return m_to_val; }

  inline void setStep(int x) { m_step = x; }
  inline int getStep() const { return m_step ; }
  void setParameters(std::map<std::string,std::string> my_keys_map);

 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODScanConfig * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);
  int fetchNextId() throw(std::runtime_error);


  // User data
  int m_ID;

  int  m_type_id;
  std::string m_type;
  int m_from_val;
  int m_to_val;
  int m_step;

};

#endif
