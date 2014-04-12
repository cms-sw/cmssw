#ifndef ODJBH4CONFIG_H
#define ODJBH4CONFIG_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IODConfig.h"

class ODJBH4Config : public IODConfig {
 public:
  friend class EcalCondDBInterface;
  ODJBH4Config();
  ~ODJBH4Config();

  // User data methods
  inline std::string getTable() { return "ECAL_JBH4_CONFIGURATION"; }

  inline void setId(int id) { m_ID = id; }
  inline int getId() const { return m_ID; }


  inline void setUseBuffer(int x) { m_use_buffer = x; }
  inline int getUseBuffer() const { return m_use_buffer; }

  inline void setHalModuleFile(std::string x) { m_hal_mod_file = x; }
  inline std::string getHalModuleFile() const { return m_hal_mod_file; }

  inline void setHalAddressTableFile(std::string x) { m_hal_add_file = x; }
  inline std::string getHalAddressTableFile() const { return m_hal_add_file; }

  inline void setHalStaticTableFile(std::string x) { m_hal_tab_file = x; }
  inline std::string getHalStaticTableFile() const { return m_hal_tab_file; }


  inline void setCbd8210SerialNumber(std::string x) { m_serial = x; }
  inline std::string getCbd8210SerialNumber() const { return m_serial; }

  inline void setCaenBridgeType(std::string x) { m_caen1 = x; }
  inline std::string getCaenBridgeType() const { return m_caen1; }

  inline void setCaenLinkNumber(int x) { m_caen2 = x; }
  inline int getCaenLinkNumber() const { return m_caen2; }

  inline void setCaenBoardNumber(int x) { m_caen3 = x ; }
  inline int getCaenBoardNumber() const { return m_caen3 ; }


 private:
  void prepareWrite()  throw(std::runtime_error);
  void writeDB()       throw(std::runtime_error);
  void clear();
  void fetchData(ODJBH4Config * result)     throw(std::runtime_error);
  int fetchID()  throw(std::runtime_error);
  int fetchNextId() throw(std::runtime_error);


  // User data
  int m_ID;

  int  m_use_buffer;
  std::string m_hal_mod_file;
  std::string m_hal_add_file;
  std::string m_hal_tab_file;
  std::string m_serial;
  std::string m_caen1;
  int m_caen2;
  int m_caen3;

};

#endif
