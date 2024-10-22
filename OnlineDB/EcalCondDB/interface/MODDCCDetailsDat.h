#ifndef MODDCCDETAILSDAT_H
#define MODDCCDETAILSDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MODRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MODDCCDetailsDat : public IDataItem {
public:
  friend class EcalCondDBInterface;
  MODDCCDetailsDat();
  ~MODDCCDetailsDat() override;

  // User data methods
  inline std::string getTable() override { return "OD_DCC_DETAILS_DAT"; }

  inline void setQPLL(int x) { m_qpll = x; }
  inline int getQPLL() const { return m_qpll; }

  inline void setOpticalLink(int x) { m_opto = x; }
  inline int getOpticalLink() const { return m_opto; }

  inline void setDataTimeout(int x) { m_tout = x; }
  inline int getDataTimeout() const { return m_tout; }

  inline void setHeader(int x) { m_head = x; }
  inline int getHeader() const { return m_head; }

  inline void setEventNumber(int x) { m_evnu = x; }
  inline int getEventNumber() const { return m_evnu; }

  inline void setBXNumber(int x) { m_bxnu = x; }
  inline int getBXNumber() const { return m_bxnu; }

  inline void setEvenParity(int x) { m_evpa = x; }
  inline int getEvenParity() const { return m_evpa; }

  inline void setOddParity(int x) { m_odpa = x; }
  inline int getOddParity() const { return m_odpa; }

  inline void setBlockSize(int x) { m_blsi = x; }
  inline int getBlockSize() const { return m_blsi; }

  inline void setAlmostFullFIFO(int x) { m_alff = x; }
  inline int getAlmostFullFIFO() const { return m_alff; }

  inline void setFullFIFO(int x) { m_fuff = x; }
  inline int getFullFIFO() const { return m_fuff; }

  inline void setForcedFullSupp(int x) { m_fusu = x; }
  inline int getForcedFullSupp() const { return m_fusu; }

private:
  void prepareWrite() noexcept(false) override;

  void writeDB(const EcalLogicID* ecid, const MODDCCDetailsDat* item, MODRunIOV* iov) noexcept(false);

  void writeArrayDB(const std::map<EcalLogicID, MODDCCDetailsDat>* data, MODRunIOV* iov) noexcept(false);

  void fetchData(std::map<EcalLogicID, MODDCCDetailsDat>* fillMap, MODRunIOV* iov) noexcept(false);

  // User data
  int m_qpll;
  int m_opto;
  int m_tout;
  int m_head;
  int m_evnu;
  int m_bxnu;
  int m_evpa;
  int m_odpa;
  int m_blsi;
  int m_alff;
  int m_fuff;
  int m_fusu;
};

#endif
