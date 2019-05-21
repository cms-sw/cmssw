#ifndef CALIIOV_H
#define CALIIOV_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/CaliTag.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

typedef int run_t;

class CaliIOV : public IIOV {
public:
  friend class EcalCondDBInterface;

  CaliIOV();
  ~CaliIOV() override;

  // Methods for user data

  void setSince(const Tm& since);
  Tm getSince() const;
  void setTill(const Tm& till);
  Tm getTill() const;
  void setCaliTag(const CaliTag& tag);
  CaliTag getCaliTag() const;

  // Methods from IUniqueDBObject
  int getID() { return m_ID; };
  int fetchID() noexcept(false) override;
  void setByID(int id) noexcept(false) override;

  // Operators
  inline bool operator==(const CaliIOV& m) const {
    return (m_caliTag == m.m_caliTag && m_since == m.m_since && m_till == m.m_till);
  }

  inline bool operator!=(const CaliIOV& m) const { return !(*this == m); }

private:
  // User data for this IOV
  Tm m_since;
  Tm m_till;
  CaliTag m_caliTag;

  int writeDB() noexcept(false);
  void setByTm(CaliTag* tag, const Tm& time) noexcept(false);
};

#endif
