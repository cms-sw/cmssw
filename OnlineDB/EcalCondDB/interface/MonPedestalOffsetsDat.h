#ifndef MONPEDESTALOFFSETSDAT_H
#define MONPEDESTALOFFSETSDAT_H

#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDataItem.h"
#include "OnlineDB/EcalCondDB/interface/MonRunTag.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class MonPedestalOffsetsDat : public IDataItem {
 public:
  friend class EcalCondDBInterface;
  MonPedestalOffsetsDat();
  ~MonPedestalOffsetsDat();

  // User data methods
  inline std::string getTable() { return "MON_PEDESTAL_OFFSETS_DAT"; }

  inline void setDACG1(int dac) { m_dacG1 = dac; }
  inline int getDACG1() const { return m_dacG1; }

  inline void setDACG6(int dac) { m_dacG6 = dac; }
  inline int getDACG6() const { return m_dacG6; }

  inline void setDACG12(int dac) { m_dacG12 = dac; }
  inline int getDACG12() const { return m_dacG12; }

  inline void setTaskStatus(bool status) { m_taskStatus = status; }
  inline bool getTaskStatus() const { return m_taskStatus; }

 private:
  void prepareWrite() 
    throw(std::runtime_error);

  void writeDB(const EcalLogicID* ecid, const MonPedestalOffsetsDat* item, MonRunIOV* iov )
    throw(std::runtime_error);

  void writeArrayDB(const std::map< EcalLogicID, MonPedestalOffsetsDat >* data, MonRunIOV* iov)
    throw(std::runtime_error);


  void fetchData(std::map< EcalLogicID, MonPedestalOffsetsDat >* fillMap, MonRunIOV* iov, std::string mappa=" cv.maps_to " )
     throw(std::runtime_error);

  // User data
  int m_dacG1;
  int m_dacG6;
  int m_dacG12;
  bool m_taskStatus;
};

#endif
