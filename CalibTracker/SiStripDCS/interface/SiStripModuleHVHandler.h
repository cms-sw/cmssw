#ifndef SISTRIPMODULEHV_SRC_HANDLER_H
#define SISTRIPMODULEHV_SRC_HANDLER_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/DataRecord/interface/SiStripModuleHVRcd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"

namespace popcon{
  class SiStripModuleHVHandler : public popcon::PopConSourceHandler<SiStripModuleHV>
    {
    public:
      void getNewObjects();
      ~SiStripModuleHVHandler();
      SiStripModuleHVHandler(const edm::ParameterSet& pset);
      
    private:
      void setForTransfer();
      bool isTransferNeeded();
      std::string m_name;
      unsigned long long m_since;
      SiStripModuleHV* SiStripModuleHV_;
      edm::Service<SiStripModuleHVBuilder> modHVBuilder;
    };
}
#endif

