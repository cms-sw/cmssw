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
  ~DCUIOV();

  // Methods for user data
  void setSince(Tm since);
  Tm getSince() const;
  void setTill(Tm till);
  Tm getTill() const;
  void setDCUTag(DCUTag tag);
  DCUTag getDCUTag() const;

  // Methods from IUniqueDBObject
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

 private:
  // User data for this IOV
  Tm m_since;
  Tm m_till;
  DCUTag m_dcuTag;

  int writeDB() throw(std::runtime_error);
  void setByTm(DCUTag* tag, Tm time) throw(std::runtime_error);
};

#endif
