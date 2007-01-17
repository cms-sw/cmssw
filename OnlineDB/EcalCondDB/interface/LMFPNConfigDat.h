#ifndef LMFPNCONFIGDAT_H
#define LMFPNCONFIGDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunTag.h"
#include "OnlineDB/EcalCondDB/interface/LMFRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LMFPNConfigDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  LMFPNConfigDat();
  ~LMFPNConfigDat();

  // User data methods
  inline std::string getTable() { return "LMF_PN_CONFIG_DAT"; }

  inline void setPNAID(int logic_id) { m_pnAID = logic_id; }
  inline int getPNAID() const { return m_pnAID; }

  inline void setPNBID(int logic_id) { m_pnBID = logic_id; }
  inline int getPNBID() const { return m_pnBID; }

  inline void setPNAValidity(bool valid) { m_pnAValidity = valid; }
  inline bool getPNAValidity() const { return m_pnAValidity; }

  inline void setPNBValidity(bool valid) { m_pnBValidity = valid; }
  inline bool getPNBValidity() const { return m_pnBValidity; }

  inline void setPNMeanValidity(bool valid) { m_pnMeanValidity = valid; }
  inline bool getPNMeanValidity() const { return m_pnMeanValidity; }
  
 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const LMFPNConfigDat* item, LMFRunIOV* iov)
    throw(std::runtime_error);

  void fetchData(std::map< EcalLogicID, LMFPNConfigDat >* fillVec, LMFRunIOV* iov)
     throw(std::runtime_error);

  // User data
  int m_pnAID;
  int m_pnBID;
  bool m_pnAValidity;
  bool m_pnBValidity;
  bool m_pnMeanValidity;

  
};

#endif
