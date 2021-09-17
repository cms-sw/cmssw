#ifndef DataFormats_Provenance_LuminosityBlockAuxiliary_h
#define DataFormats_Provenance_LuminosityBlockAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// Auxiliary luminosity block data that is persistent

namespace edm {
  class LuminosityBlockAux;
  class LuminosityBlockAuxiliary {
  public:
    friend void conversion(LuminosityBlockAux const&, LuminosityBlockAuxiliary&);
    LuminosityBlockAuxiliary() : processHistoryID_(), id_(), beginTime_(), endTime_() {}
    LuminosityBlockAuxiliary(LuminosityBlockID const& theId, Timestamp const& theTime, Timestamp const& theEndTime)
        : processHistoryID_(), id_(theId), beginTime_(theTime), endTime_(theEndTime) {}
    LuminosityBlockAuxiliary(RunNumber_t const& theRun,
                             LuminosityBlockNumber_t const& theLumi,
                             Timestamp const& theTime,
                             Timestamp const& theEndTime)
        : processHistoryID_(), id_(theRun, theLumi), beginTime_(theTime), endTime_(theEndTime) {}
    LuminosityBlockAuxiliary(LuminosityBlockAuxiliary&&) = default;
    LuminosityBlockAuxiliary(LuminosityBlockAuxiliary const&) = default;
    ~LuminosityBlockAuxiliary() {}
    LuminosityBlockAuxiliary& operator=(LuminosityBlockAuxiliary&&) = default;
    LuminosityBlockAuxiliary& operator=(LuminosityBlockAuxiliary const&) = default;
    void write(std::ostream& os) const;
    ProcessHistoryID const& processHistoryID() const { return processHistoryID_; }
    void setProcessHistoryID(ProcessHistoryID const& phid) { processHistoryID_ = phid; }
    LuminosityBlockNumber_t luminosityBlock() const { return id().luminosityBlock(); }
    RunNumber_t run() const { return id().run(); }
    LuminosityBlockID const& id() const { return id_; }
    LuminosityBlockID& id() { return id_; }
    Timestamp const& beginTime() const { return beginTime_; }
    void setBeginTime(Timestamp const& time) {
      if (beginTime_ == Timestamp::invalidTimestamp())
        beginTime_ = time;
    }
    Timestamp const& endTime() const { return endTime_; }
    void setEndTime(Timestamp const& time) {
      if (endTime_ == Timestamp::invalidTimestamp())
        endTime_ = time;
    }
    void mergeAuxiliary(LuminosityBlockAuxiliary const& newAux);

    bool sameIdentity(LuminosityBlockAuxiliary const& iRHS) const {
      return iRHS.processHistoryID_ == processHistoryID_ && iRHS.id_ == id_;
    }

  private:
    // This is the ID of the full process history (not the reduced process history).
    // In cases where LuminosityBlock's are merged, the ID of the first process history encountered
    // is stored here.
    ProcessHistoryID processHistoryID_;
    // LuminosityBlock ID
    LuminosityBlockID id_;
    // Times from DAQ
    Timestamp beginTime_;
    Timestamp endTime_;
  };

  inline std::ostream& operator<<(std::ostream& os, const LuminosityBlockAuxiliary& p) {
    p.write(os);
    return os;
  }

}  // namespace edm
#endif
