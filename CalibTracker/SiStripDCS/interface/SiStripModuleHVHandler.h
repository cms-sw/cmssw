#ifndef SISTRIPMODULEHV_SRC_HANDLER_H
#define SISTRIPMODULEHV_SRC_HANDLER_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/DataRecord/interface/SiStripModuleHVRcd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"

namespace popcon{
class SiStripModuleHVHandler : public popcon::PopConSourceHandler<SiStripModuleHV>
{
    public:
        void getNewObjects();
        ~SiStripModuleHVHandler();
        SiStripModuleHVHandler(std::string,std::string,std::string, const edm::Event& evt, const edm::EventSetup& est, std::string);
    private:
        SiStripModuleHV* SiStripModuleHV_;

};

}
#endif

