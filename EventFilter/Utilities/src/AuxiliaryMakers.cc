#include "EventFilter/Utilities/interface/AuxiliaryMakers.h"
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
	edm::Timestamp tstamp = edm::Timestamp(static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl));
	return edm::EventAuxiliary(eventId,
				   processGUID,
				   tstamp,
				   true,
				   (edm::EventAuxiliary::ExperimentType)record->getHistory().history().hist[0].eventtype,
				   (int)record->getHeader().getData().header.bcid,
				   edm::EventAuxiliary::invalidStoreNumber,
				   record->getHeader().getData().header.orbitLow);
      }
  }
}

