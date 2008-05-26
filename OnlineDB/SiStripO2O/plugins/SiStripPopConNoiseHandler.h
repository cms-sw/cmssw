#ifndef SISTRIPPOPCON_NOISE_HANDLER_H
#define SISTRIPPOPCON_NOISE_HANDLER_H

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

namespace popcon{
  class SiStripPopConNoiseHandler : public popcon::PopConSourceHandler<SiStripNoises>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~SiStripPopConNoiseHandler(); 
    SiStripPopConNoiseHandler(const edm::ParameterSet& pset); 
    
  private:

    bool isTransferNeeded();
    void setForTransfer();

    std::string m_name;
    unsigned long long m_since;
    bool m_debugMode;
    edm::Service<SiStripCondObjBuilderFromDb> condObjBuilder;
  };
}
#endif // SISTRIPPOPCON_NOISE_HANDLER_H
