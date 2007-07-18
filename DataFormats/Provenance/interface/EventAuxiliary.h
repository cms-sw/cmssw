#ifndef DataFormats_Provenance_EventAuxiliary_h
#define DataFormats_Provenance_EventAuxiliary_h

#include <iosfwd>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"

// Auxiliary event data that is persistent

namespace edm
{
  struct EventAuxiliary {
    EventAuxiliary() :
	processHistoryID_(),
	id_(),
	time_(),
	luminosityBlock_(),
	isRealData_(false), 
	experimentType_(checkExperimentType(std::string("DAQ"))) {}
    EventAuxiliary(EventID const& theId, Timestamp const& theTime, LuminosityBlockNumber_t lb,
                     bool isReal, std::string eType = std::string("DAQ")) :
	processHistoryID_(),
	id_(theId),
	time_(theTime),
	luminosityBlock_(lb),
	isRealData_(isReal),
        experimentType_(checkExperimentType(eType)) {}
    ~EventAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    EventID const& id() const {return id_;}
    Timestamp const& time() const {return time_;}
    LuminosityBlockNumber_t const& luminosityBlock() const {return luminosityBlock_;}
    EventNumber_t event() const {return id_.event();}
    RunNumber_t run() const {return id_.run();}
    bool isRealData() const {return isRealData_;}
    std::string experimentType() const {return experimentType_;}

    std::string & checkExperimentType(const std::string &);

// most recently process that processed this event
// is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
// Event ID
    EventID                  id_;
// Time from DAQ.
    Timestamp                time_;
// Associated Luminosity Block identifier.
    LuminosityBlockNumber_t  luminosityBlock_;
// Is this real data (i.e. not simulated).
    bool                     isRealData_;
// Something descriptive of the source of the data.
    std::string              experimentType_;

    typedef std::vector<std::string> svec;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const EventAuxiliary& p) {
    p.write(os);
    return os;
  }
}

#endif
