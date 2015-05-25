#include <sys/time.h>

#include "EventFilter/Utilities/interface/AuxiliaryMakers.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace evf{
  namespace evtn{

      edm::EventAuxiliary makeEventAuxiliary(TCDSRecord *record,
					     unsigned int runNumber,
                                             unsigned int lumiSection,
					     std::string const &processGUID){
	edm::EventID eventId(runNumber, // check that runnumber from record is consistent
			     //record->getHeader().getData().header.lumiSection,//+1
                             lumiSection,
			     record->getHeader().getData().header.eventNumber);

	uint64_t gpsh = record->getBST().getBST().gpstimehigh;
	uint32_t gpsl = record->getBST().getBST().gpstimelow;
        edm::TimeValue_t time = static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl);
        if (time == 0) {
          timeval stv;
          gettimeofday(&stv,0);
          time = stv.tv_sec;
          time = (time << 32) + stv.tv_usec;
        }
	uint64_t orbitnr = (((uint64_t)record->getHeader().getData().header.orbitHigh) << 16) + record->getHeader().getData().header.orbitLow;
        uint32_t recordLumiSection = record->getHeader().getData().header.lumiSection;

        if (recordLumiSection != lumiSection) 
          edm::LogWarning("AuxiliaryMakers") << "Lumisection mismatch, external : "<<lumiSection << ", record : " << recordLumiSection; 
        if ((orbitnr >> 18) + 1 != recordLumiSection)
          edm::LogWarning("AuxiliaryMakers") << "Lumisection and orbit number mismatch, LS : " << lumiSection << ", LS from orbit: " << ((orbitnr >> 18) + 1) << ", orbit:" << orbitnr;

	return edm::EventAuxiliary(eventId,
				   processGUID,
				   edm::Timestamp(time),
				   true,
                                   (edm::EventAuxiliary::ExperimentType)(FED_EVTY_EXTRACT(record->getFEDHeader().getData().header.eventid)),
				   (int)record->getHeader().getData().header.bcid,
				   edm::EventAuxiliary::invalidStoreNumber,
				   (int)(orbitnr&0x7fffffffU));//framework supports only 32-bit signed
      }
  }
}
