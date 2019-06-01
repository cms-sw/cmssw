#ifndef ODVFETOREJECTDAT_H
#define ODVFETOREJECTDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODVfeToRejectInfo.h"

class ODVfeToRejectDat : public IODConfig {
public:
  friend class EcalCondDBInterface;
  ODVfeToRejectDat();
  ~ODVfeToRejectDat() override;

  // User data methods
  inline std::string getTable() override { return "VFES_TO_REJECT_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setVfeId(int dac) { m_vfe = dac; }
  inline int getVfeId() const { return m_vfe; }

  inline void setGain(int dac) { m_gain = dac; }
  inline int getGain() const { return m_gain; }

  inline void setStatus(int dac) { m_sta = dac; }
  inline int getStatus() const { return m_sta; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const ODVfeToRejectDat* item, ODVfeToRejectInfo* iov) noexcept(false);

  void writeArrayDB(const std::vector<ODVfeToRejectDat>& data, ODVfeToRejectInfo* iov) noexcept(false);

  void fetchData(std::vector<ODVfeToRejectDat>* fillMap, ODVfeToRejectInfo* iov) noexcept(false);

  // User data

  int m_fed;
  int m_tt;
  int m_vfe;
  int m_gain;
  int m_sta;
  int m_ID;
};

#endif
