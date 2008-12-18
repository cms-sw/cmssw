#ifndef DataFormats_Provenance_LuminosityBlockAuxiliary_h
#define DataFormats_Provenance_LuminosityBlockAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary luminosity block data that is persistent

namespace edm
{
  struct LuminosityBlockAuxiliary {
    LuminosityBlockAuxiliary() :
	processHistoryID_(),
	id_(),
	beginTime_(),
	endTime_() {}
    LuminosityBlockAuxiliary(LuminosityBlockID const& theId,
			     Timestamp const& theTime,
			     Timestamp const& theEndTime) :
	processHistoryID_(),
	id_(theId),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    LuminosityBlockAuxiliary(RunNumber_t const& theRun,
			     LuminosityBlockNumber_t const& theLumi,
			     Timestamp const& theTime,
			     Timestamp const& theEndTime) :
	processHistoryID_(),
	id_(theRun, theLumi),
	beginTime_(theTime),
	endTime_(theEndTime) {}
    ~LuminosityBlockAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    void setProcessHistoryID(ProcessHistoryID const& phid) const {processHistoryID_ = phid;}
    LuminosityBlockNumber_t luminosityBlock() const {return id().luminosityBlock();}
    RunNumber_t run() const {return id().run();}
    LuminosityBlockID const& id() const {return id_;}
    Timestamp const& beginTime() const {return beginTime_;}
    Timestamp const& endTime() const {return endTime_;}
    void setEndTime(Timestamp const& time) {
      if (endTime_ == Timestamp::invalidTimestamp()) endTime_ = time;
    }
    bool mergeAuxiliary(LuminosityBlockAuxiliary const& newAux);

    // most recent process that processed this lumi block
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // LuminosityBlock ID
    LuminosityBlockID id_;
    // Times from DAQ
    Timestamp beginTime_;
    Timestamp endTime_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const LuminosityBlockAuxiliary& p) {
    p.write(os);
    return os;
  }

}
#endif
