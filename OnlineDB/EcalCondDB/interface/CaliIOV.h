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
  ~CaliIOV();

  // Methods for user data

  
  void setSince(Tm since);
  Tm getSince() const;
  void setTill(Tm till);
  Tm getTill() const;
  void setCaliTag(CaliTag tag);
  CaliTag getCaliTag() const;

  // Methods from IUniqueDBObject
  int getID(){ return m_ID;} ;
  int fetchID() throw(std::runtime_error);
  void setByID(int id) throw(std::runtime_error);

  // Operators
  inline bool operator==(const CaliIOV &m) const
    {
      return ( m_caliTag   == m.m_caliTag &&
	       m_since == m.m_since &&
	       m_till   == m.m_till );
    }

  inline bool operator!=(const CaliIOV &m) const { return !(*this == m); }


 private:
  // User data for this IOV
  Tm m_since;
  Tm m_till;
  CaliTag m_caliTag;

  int writeDB() throw(std::runtime_error);
  void setByTm(CaliTag* tag, Tm time) throw(std::runtime_error);
};

#endif
