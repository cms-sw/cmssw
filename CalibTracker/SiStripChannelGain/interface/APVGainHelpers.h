#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H


#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>
#include <vector>
#include <utility>
#include <stdint.h>



namespace APVGain {
    int subdetectorId(uint32_t);
    int subdetectorId(const char*);
    int subdetectorSide(uint32_t);
    int subdetectorSide(const char*);
    int subdetectorPlane(uint32_t);
    int subdetectorPlane(const char*);

    std::vector<MonitorElement*> FetchMonitor(std::vector<MonitorElement*>, uint32_t);
    std::vector<std::pair<std::string,std::string>> monHnames(std::vector<std::string>,bool,const char* tag);
};

#endif
