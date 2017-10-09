#ifndef FILLINFOPOPCONSOURCEHANDLER_H
#define FILLINFOPOPCONSOURCEHANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/FillInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class FillInfoPopConSourceHandler : public popcon::PopConSourceHandler<FillInfo>{
 public:
  FillInfoPopConSourceHandler( const edm::ParameterSet& pset ); 
  ~FillInfoPopConSourceHandler();
  void getNewObjects();
  std::string id() const;
  
 private:
  bool m_debug;
  unsigned short m_firstFill, m_lastFill;
  std::string m_name;  
  //for reading from relational database source 
  std::string m_connectionString, m_dipSchema, m_authpath;
};

#endif 
