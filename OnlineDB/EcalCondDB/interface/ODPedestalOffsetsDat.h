#ifndef ODPEDESTALOFFSETSDAT_H
#define ODPEDESTALOFFSETSDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/ODFEPedestalOffsetInfo.h"

class ODPedestalOffsetsDat : public IODConfig {
public:
  friend class EcalCondDBInterface;
  ODPedestalOffsetsDat();
  ~ODPedestalOffsetsDat() override;

  // User data methods
  inline std::string getTable() override { return "PEDESTAL_OFFSETS_DAT"; }

  inline void setId(int dac) { m_ID = dac; }
  inline int getId() const { return m_ID; }

  inline void setSMId(int dac) { m_sm = dac; }
  inline int getSMId() const { return m_sm; }

  inline void setFedId(int dac) { m_fed = dac; }
  inline int getFedId() const { return m_fed; }

  inline void setTTId(int dac) { m_tt = dac; }
  inline int getTTId() const { return m_tt; }

  inline void setCrystalId(int dac) { m_xt = dac; }
  inline int getCrystalId() const { return m_xt; }

  inline void setLow(int dac) { m_low = dac; }
  inline int getLow() const { return m_low; }

  inline void setMid(int dac) { m_mid = dac; }
  inline int getMid() const { return m_mid; }

  inline void setHigh(int dac) { m_high = dac; }
  inline int getHigh() const { return m_high; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const ODPedestalOffsetsDat* item, ODFEPedestalOffsetInfo* iov) noexcept(false);

  void writeArrayDB(const std::vector<ODPedestalOffsetsDat>& data, ODFEPedestalOffsetInfo* iov) noexcept(false);

  void fetchData(std::vector<ODPedestalOffsetsDat>* fillMap, ODFEPedestalOffsetInfo* iov) noexcept(false);

  // User data
  int m_sm;
  int m_fed;
  int m_tt;
  int m_xt;
  int m_low;
  int m_mid;
  int m_high;
  int m_ID;
};

#endif
