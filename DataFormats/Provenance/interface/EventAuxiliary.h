#ifndef DataFormats_Provenance_EventAuxiliary_h
#define DataFormats_Provenance_EventAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary event data that is persistent

namespace edm {
  class EventAux;
  class EventAuxiliary {
  public:
    friend void conversion(EventAux const&, EventAuxiliary&);
    // Updated on 9 Feb. '09 on a request from Emelio Meschi
    enum ExperimentType {
      Undefined          =  0,
      PhysicsTrigger     =  1,
      CalibrationTrigger =  2,
      RandomTrigger      =  3,
      Reserved           =  4, 
      TracedEvent        =  5,
      TestTrigger        =  6,
      ErrorTrigger       = 15
    };
    static int const invalidBunchXing = -1;
    static int const invalidStoreNumber = 0;
    EventAuxiliary() :
	processHistoryID_(),
	id_(),
        processGUID_(),
	time_(),
	luminosityBlock_(0U),
	isRealData_(false), 
	experimentType_(Undefined),
	bunchCrossing_(invalidBunchXing),
	orbitNumber_(invalidBunchXing),
        storeNumber_(invalidStoreNumber) {}
    EventAuxiliary(EventID const& theId, std::string const& theProcessGUID, Timestamp const& theTime,
		   bool isReal, ExperimentType eType = Undefined,
		   int bunchXing = invalidBunchXing, int storeNumber = invalidStoreNumber,
                   int orbitNum = invalidBunchXing) :
	processHistoryID_(),
	id_(theId),
        processGUID_(theProcessGUID),
	time_(theTime),
	luminosityBlock_(0U),
	isRealData_(isReal),
        experimentType_(eType),
	bunchCrossing_(bunchXing),
	orbitNumber_(orbitNum),
	storeNumber_(storeNumber) {}
    ~EventAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    EventID const& id() const {return id_;}
    EventID& id() {return id_;}
    std::string const& processGUID() const {return processGUID_;}
    Timestamp const& time() const {return time_;}
    LuminosityBlockNumber_t luminosityBlock() const {return id_.luminosityBlock() != 0U ? id_.luminosityBlock() : luminosityBlock_;}
    void resetObsoleteInfo() {luminosityBlock_ = 0;}
    EventNumber_t event() const {return id_.event();}
    RunNumber_t run() const {return id_.run();}
    bool isRealData() const {return isRealData_;}
    ExperimentType experimentType() const {return experimentType_;}
    int bunchCrossing() const {return bunchCrossing_;}
    int orbitNumber() const {return orbitNumber_;}
    int storeNumber() const {return storeNumber_;}

  private:
    // most recently process that processed this event
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // Event ID
    EventID id_;
    // Globally unique process ID of process that created event.
    std::string processGUID_;
    // Time from DAQ
    Timestamp time_;
    // Associated Luminosity Block identifier (obsolete. for backward compatibility only)
    LuminosityBlockNumber_t luminosityBlock_;
    // Is this real data (i.e. not simulated)
    bool isRealData_;
    // Something descriptive of the source of the data
    ExperimentType experimentType_;
    //  The bunch crossing number
    int bunchCrossing_;
    // The orbit number
    int orbitNumber_;
    //  The LHC store number
    int storeNumber_;
  };

  bool
  isSameEvent(EventAuxiliary const& a, EventAuxiliary const& b);

  inline
  std::ostream&
  operator<<(std::ostream& os, const EventAuxiliary& p) {
    p.write(os);
    return os;
  }

}

#endif
