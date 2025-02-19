#ifndef DCSPTMTEMP_H
#define DCSPTMTEMP_H

#include <stdexcept>
#include <iostream>

#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"


typedef int run_t;

class DCSPTMTemp  : public IDBObject {
 public:
  friend class EcalCondDBInterface;

  DCSPTMTemp();
  ~DCSPTMTemp();

  // Methods for user data
  inline std::string getTable() { return ""; }
  
  float getTemperature() ;
  void setTemperature(float temp);

  void setStart(Tm start);
  Tm getStart() const;
  void setEnd(Tm end);
  Tm getEnd() const;
  EcalLogicID getEcalLogicID() const;
  void setEcalLogicID(EcalLogicID ecid);

  
 private:
  // User data
  int m_ID;
  EcalLogicID m_ecid;
  float m_temp;
  Tm m_runStart;
  Tm m_runEnd;


};

#endif
