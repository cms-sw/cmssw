#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_APVGAINHELPERS_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      
    public:

    APVmon(int v1, int v2, int v3, MonitorElement* v4) :
      m_subdetectorId(v1),m_subdetectorSide(v2),m_subdetectorPlane(v3),m_monitor(v4){}

      int getSubdetectorId(){
	return m_subdetectorId;
      }

      int getSubdetectorSide(){
	return m_subdetectorSide;
      }

      int getSubdetectorPlane(){
	return m_subdetectorPlane;
      }

      MonitorElement* getMonitor(){
	return m_monitor;
      }

      void printAll(){
	LogDebug("APVGainHelpers")<< "subDetectorID:" << m_subdetectorId << std::endl;
	LogDebug("APVGainHelpers")<< "subDetectorSide:" << m_subdetectorSide << std::endl;
	LogDebug("APVGainHelpers")<< "subDetectorPlane:" << m_subdetectorPlane << std::endl;
	LogDebug("APVGainHelpers")<< "histoName:" << m_monitor->getName() << std::endl;
	return;
      }

    private:

      int m_subdetectorId;
      int m_subdetectorSide;
      int m_subdetectorPlane;
      MonitorElement* m_monitor;

    };

    std::vector<MonitorElement*> FetchMonitor(std::vector<APVmon>, uint32_t, const TrackerTopology* topo=nullptr);
};

#endif
