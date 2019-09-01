#ifndef FECONFLINDAT_H
#define FECONFLINDAT_H

#include <vector>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/FEConfigLinInfo.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class FEConfigLinDat : public IDataItem {
public:
  friend class EcalCondDBInterface;  // XXX temp should not need
  FEConfigLinDat();
  ~FEConfigLinDat() override;

  // User data methods
  inline std::string getTable() override { return "FE_CONFIG_LIN_DAT"; }

  inline void setMultX12(int x) { m_multx12 = x; }
  inline void setMultX6(int x) { m_multx6 = x; }
  inline void setMultX1(int x) { m_multx1 = x; }
  inline void setShift12(int x) { m_shift12 = x; }
  inline void setShift6(int x) { m_shift6 = x; }
  inline void setShift1(int x) { m_shift1 = x; }

  inline int getMultX12() const { return m_multx12; }
  inline int getMultX6() const { return m_multx6; }
  inline int getMultX1() const { return m_multx1; }
  inline int getShift12() const { return m_shift12; }
  inline int getShift6() const { return m_shift6; }
  inline int getShift1() const { return m_shift1; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const FEConfigLinDat* item, FEConfigLinInfo* iconf) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, FEConfigLinDat>* data, FEConfigLinInfo* iconf) noexcept(false);

  void fetchData(std::map<EcalLogicID, FEConfigLinDat>* fillMap, FEConfigLinInfo* iconf) noexcept(false);

  // User data
  int m_multx12;
  int m_multx6;
  int m_multx1;
  int m_shift12;
  int m_shift6;
  int m_shift1;
};

#endif
