#ifndef FILLINFOPOPCONSOURCEHANDLER_H
#define FILLINFOPOPCONSOURCEHANDLER_H

#include <string>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RunInfo/interface/FillInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class FillInfoPopConSourceHandler : public popcon::PopConSourceHandler<FillInfo>{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~FillInfoPopConSourceHandler();
  FillInfoPopConSourceHandler(const edm::ParameterSet& pset); 
  
 private:
  bool m_debug;
  unsigned short m_fill;
  std::string m_name;  
  // for reading from omds 
  std::string m_connectionString; 
  std::string m_authpath;
};

#endif 
