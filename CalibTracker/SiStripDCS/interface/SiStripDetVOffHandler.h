#ifndef SISTRIPDETVOFF_SRC_HANDLER_H
#define SISTRIPDETVOFF_SRC_HANDLER_H

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"


namespace popcon{
  class SiStripDetVOffHandler : public popcon::PopConSourceHandler<SiStripDetVOff>
  {
  public:
    void getNewObjects();
    ~SiStripDetVOffHandler();
    SiStripDetVOffHandler(const edm::ParameterSet& pset);
    std::string id() const { return name_;}
      
  private:
    void setUserTextLog();
    void setForTransfer();
    std::string name_;
    std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > resultVec;
    edm::Service<SiStripDetVOffBuilder> modHVBuilder;
    uint32_t deltaTmin_;
    uint32_t maxIOVlength_;
    bool debug_;
  };
}
#endif

