#ifndef CALIGAINRATIODAT_H
#define CALIGAINRATIODAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/CaliIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class CaliGainRatioDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  CaliGainRatioDat();
  ~CaliGainRatioDat() override;

  // User data methods
  inline std::string getTable() override { return "CALI_GAIN_RATIO_DAT"; }

  inline void setG1G12(float c) { m_g1_g12 = c; }
  inline float getG1G12() const { return m_g1_g12; }

  inline void setG6G12(float c) { m_g6_g12 = c; }
  inline float getG6G12() const { return m_g6_g12; }

  inline void setTaskStatus(bool s) { m_taskStatus = s; }
  inline bool getTaskStatus() const { return m_taskStatus; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const CaliGainRatioDat* item, CaliIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, CaliGainRatioDat>* fillVec, CaliIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, CaliGainRatioDat>* data, CaliIOV* iov) noexcept(false);

  // User data
  float m_g1_g12;
  float m_g6_g12;
  bool m_taskStatus;
};

#endif
