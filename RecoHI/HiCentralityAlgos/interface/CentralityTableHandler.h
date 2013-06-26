#ifndef __CentralityPopCon_h__ 
#define __CentralityPopCon_h__

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
//#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class CentralityTableHandler : public popcon::PopConSourceHandler<CentralityTable>
{
 public:
  CentralityTableHandler(edm::ParameterSet const & pset):
    //    popcon::PopConSourceHandler<CentralityTable>(pset),
    inputTFileName_(pset.getParameter<std::string>("inputFile")),
    centralityTag_(pset.getParameter<std::string>("rootTag"))
      {;}
    ~CentralityTableHandler(){;}
    void getNewObjects();
    std::string id() const{return "CentralityTableHandler";}
    
 private:  
  std::string inputTFileName_;
  std::string centralityTag_;
  int runnum_;
};

#endif








