#ifndef EVENTFILTER_UTILITIES_PLUGINS_EVFFEDFILLEREP
#define EVENTFILTER_UTILITIES_PLUGINS_EVFFEDFILLEREP

#include <unistd.h>

#include "EventFilter/FEDInterface/interface/FED1023.h"

namespace evf{

  class EvffedFillerEP{

  public:
    EvffedFillerEP(){
    }
    void setEPTimeStamp(uint64_t ts, unsigned char *payload){
      *(uint64_t*) (payload + fedinterface::EVFFED_EPWCTIM_OFFSET) = ts;
    }
    void setEPProcessId(pid_t pid, unsigned char *payload){
      *(uint32_t*)(payload+fedinterface::EVFFED_EPIDENT_OFFSET) = 
	(pid & fedinterface::EVFFED_EPPCIDE_MASK) << fedinterface::EVFFED_EPPCIDE_SHIFT;
    }
    void setEPEventId(uint32_t eid, unsigned char *payload){
      *(uint32_t*)(payload+fedinterface::EVFFED_EPEVENT_OFFSET) =
	eid;
    }
    void setEPEventCount(uint32_t evc, unsigned char *payload){
      *(uint32_t*)(payload+fedinterface::EVFFED_EPEVTCT_OFFSET) =
	evc;
    }
    void setEPEventHisto(uint64_t ehi, unsigned char *payload){
      *(uint64_t*)(payload+fedinterface::EVFFED_EPHISTO_OFFSET) =
	ehi;
    }

  private:
    
  };
}


#endif
