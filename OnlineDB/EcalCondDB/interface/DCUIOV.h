#ifndef DCUIOV_H
#define DCUIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/DCUTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int run_t;

class DCUIOV : public IIOV {
public:
  friend class EcalCondDBInterface;

  DCUIOV();
  ~DCUIOV() override;

  // Methods for user data
  void setSince(const Tm& since);
  Tm getSince() const;
  void setTill(const Tm& till);
  Tm getTill() const;
  void setDCUTag(const DCUTag& tag);
  DCUTag getDCUTag() const;

  // Methods from IUniqueDBObject
  int getID() { return m_ID; };
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators
  inline bool operator==(const DCUIOV& m) const {
    return (m_dcuTag == m.m_dcuTag && m_since == m.m_since && m_till == m.m_till);
  }

  inline bool operator!=(const DCUIOV& m) const { return !(*this == m); }

private:
  // User data for this IOV
  Tm m_since;
  Tm m_till;
  DCUTag m_dcuTag;

  int writeDB() noexcept(false);
  void setByTm(DCUTag* tag, const Tm& time) noexcept(false);
};

#endif
