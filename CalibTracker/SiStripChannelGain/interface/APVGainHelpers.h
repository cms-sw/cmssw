#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>
#include <vector>
#include <utility>
#include <cstdint>


namespace APVGain {
    int subdetectorId(uint32_t);
    int subdetectorId(const std::string&);
    int subdetectorSide(uint32_t,const TrackerTopology*);
    int subdetectorSide(const std::string&);
    int subdetectorPlane(uint32_t,const TrackerTopology*);
    int subdetectorPlane(const std::string&);

    std::vector<std::pair<std::string,std::string>> monHnames(std::vector<std::string>,bool,const char* tag);

    struct APVmon{
        int subdetectorId;
        int subdetectorSide;
        int subdetectorPlane;
        MonitorElement* monitor;

        APVmon(int v1, int v2, int v3, MonitorElement* v4) :
            subdetectorId(v1),subdetectorSide(v2),subdetectorPlane(v3),monitor(v4) {}
    };

    std::vector<MonitorElement*> FetchMonitor(std::vector<APVmon>, uint32_t, const TrackerTopology* topo=nullptr);
};

#endif
