#ifndef DataFormats_Provenance_LuminosityBlockAuxiliary_h
#define DataFormats_Provenance_LuminosityBlockAuxiliary_h

#include <iosfwd>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"

// Auxiliary luminosity block data that is persistent

namespace edm
{
  struct LuminosityBlockAuxiliary {
    LuminosityBlockAuxiliary() :
	processHistoryID_(),
	id_() {}
    LuminosityBlockAuxiliary(LuminosityBlockID const& theID) :
	processHistoryID_(),
	id_(theID) {}
    LuminosityBlockAuxiliary(RunNumber_t const& theRun, LuminosityBlockNumber_t const& theLumi) :
	processHistoryID_(),
	id_(theRun, theLumi) {}
    ~LuminosityBlockAuxiliary() {}
    void write(std::ostream& os) const;
    ProcessHistoryID& processHistoryID() const {return processHistoryID_;}
    LuminosityBlockNumber_t luminosityBlock() const {return id().luminosityBlock();}
    RunNumber_t run() const {return id().run();}
    LuminosityBlockID const& id() const {return id_;}
    // most recent process that processed this lumi block
    // is the last on the list, this defines what "latest" is
    mutable ProcessHistoryID processHistoryID_;
    // LuminosityBlock ID
    LuminosityBlockID id_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const LuminosityBlockAuxiliary& p) {
    p.write(os);
    return os;
  }

}
#endif
