#ifndef SISTRIPMODULEHV_SRC_HANDLER_H
#define SISTRIPMODULEHV_SRC_HANDLER_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"

namespace popcon{
  class SiStripModuleHVHandler : public popcon::PopConSourceHandler<SiStripDetVOff>
    {
    public:
      void getNewObjects();
      ~SiStripModuleHVHandler();
      SiStripModuleHVHandler(const edm::ParameterSet& pset);
      std::string id() const { return m_name;}
      
    private:
      void setForTransfer();
      std::string m_name;
      std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > resultVec;
      edm::Service<SiStripModuleHVBuilder> modHVBuilder;
    };
}
#endif

