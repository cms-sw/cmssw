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
      
      APVmon(int v1, int v2, int v3, MonitorElement* v4) :
	subdetectorId(v1),
	subdetectorSide(v2),
	subdetectorPlane(v3),
	monitor(std::move(v4)) 
      {
      }

    public:

      int getSubdetectorId(){
	return subdetectorId;
      }

      int getSubdetectorSide(){
	return subdetectorSide;
      }

      int getSubdetectorPlane(){
	return subdetectorPlane;
      }

      MonitorElement* getTheMon(){
	return monitor;
      }

      void printAll(){
	std::cout<< "subDetectorID:" << subdetectorId << std::endl;
	std::cout<< "subDetectorSide:" << subdetectorSide << std::endl;
	std::cout<< "subDetectorPlane:" << subdetectorPlane << std::endl;
	std::cout<< "histoName:" << monitor->getName() << std::endl;
	return;
      }

    private:

      int subdetectorId;
      int subdetectorSide;
      int subdetectorPlane;
      MonitorElement* monitor;

    };

    std::vector<MonitorElement*> FetchMonitor(std::vector<APVmon>, uint32_t, const TrackerTopology* topo=nullptr);
};

#endif
