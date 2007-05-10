#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"

#include "OnlineDB/EcalCondDB/interface/DCSPTMTemp.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"
#include "OnlineDB/EcalCondDB/interface/DateHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

using namespace std;
using namespace oracle::occi;

DCSPTMTemp::DCSPTMTemp()
{
  m_conn = NULL;
 

  m_runStart = Tm();
  m_runEnd = Tm();
  m_temp=0.;
 
}



DCSPTMTemp::~DCSPTMTemp()
{
}




void DCSPTMTemp::setStart(Tm start)
{
  if (start != m_runStart) {
    m_ID = 0;
    m_runStart = start;
  }
}



Tm DCSPTMTemp::getStart() const
{
  return m_runStart;
}



void DCSPTMTemp::setEnd(Tm end)
{
  if (end != m_runEnd) {
    m_ID = 0;
    m_runEnd = end;
  }
}



Tm DCSPTMTemp::getEnd() const
{
  return m_runEnd;
}

float DCSPTMTemp::getTemperature() 
{
  return m_temp;
}

void DCSPTMTemp::setTemperature(float temp) 
{
  m_temp=temp;
}

EcalLogicID DCSPTMTemp::getEcalLogicID() const
{
  return m_ecid;
}

void DCSPTMTemp::setEcalLogicID(EcalLogicID ecid) 
{
  m_ecid=ecid;
}

