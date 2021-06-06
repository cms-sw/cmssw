#ifndef DataFormats_Provenance_CompactEventAuxiliaryVector_h
#define DataFormats_Provenance_CompactEventAuxiliaryVector_h

#include <vector>
#include <unordered_set>

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/Utilities/interface/hash_combine.h"

namespace edm {
  class CompactEventAuxiliaryVector {
  public:
    using ExperimentType = EventAuxiliary::ExperimentType;
    static int const invalidBunchXing = EventAuxiliary::invalidBunchXing;
    static int const invalidStoreNumber = EventAuxiliary::invalidStoreNumber;

    // These components of EventAuxiliary change infrequently, so
    // they are stored in a std::unordered_set with a reference in
    // CompactEventAuxiliary
    class CompactEventAuxiliaryExtra {
    public:
      CompactEventAuxiliaryExtra(bool isReal, ExperimentType eType, int storeNum)
          : processHistoryID_(), isRealData_(isReal), experimentType_(eType), storeNumber_(storeNum) {}
      CompactEventAuxiliaryExtra(const EventAuxiliary& ea)
          : processHistoryID_(ea.processHistoryID()),
            isRealData_(ea.isRealData()),
            experimentType_(ea.experimentType()),
            storeNumber_(ea.storeNumber()) {}

      bool operator==(const CompactEventAuxiliaryExtra& extra) const {
        return processHistoryID_ == extra.processHistoryID_ && isRealData_ == extra.isRealData_ &&
               experimentType_ == extra.experimentType_ && storeNumber_ == extra.storeNumber_;
      }
      void write(std::ostream& os) const;

      // Process history ID of the full process history (not the reduced process history)
      ProcessHistoryID processHistoryID_;
      // Is this real data (i.e. not simulated)
      bool isRealData_;
      // Something descriptive of the source of the data
      ExperimentType experimentType_;
      //  The LHC store number
      int storeNumber_;
    };

    struct ExtraHash {
      std::size_t operator()(CompactEventAuxiliaryExtra const& extra) const noexcept {
        return hash_value(
            extra.processHistoryID_.compactForm(), extra.isRealData_, extra.experimentType_, extra.storeNumber_);
      }
    };

    using GUIDmemo = std::unordered_set<std::string>;
    using extraMemo = std::unordered_set<CompactEventAuxiliaryExtra, ExtraHash>;

    class CompactEventAuxiliary {
    public:
      CompactEventAuxiliary(EventID const& theId,
                            std::string const& processGUID,
                            Timestamp const& theTime,
                            int bunchXing,
                            int orbitNum,
                            CompactEventAuxiliaryExtra const& extra,
                            GUIDmemo& guidmemo,
                            extraMemo& extramemo)
          : id_(theId),
            processGUID_(memoize(processGUID, guidmemo)),
            time_(theTime),
            bunchCrossing_(bunchXing),
            orbitNumber_(orbitNum),
            extra_(memoize(extra, extramemo)) {}
      CompactEventAuxiliary(const EventAuxiliary& ea, GUIDmemo& guidmemo, extraMemo& extramemo)
          : id_(ea.id()),
            processGUID_(memoize(ea.processGUID(), guidmemo)),
            time_(ea.time()),
            bunchCrossing_(ea.bunchCrossing()),
            orbitNumber_(ea.orbitNumber()),
            extra_(memoize(CompactEventAuxiliaryExtra(ea), extramemo)) {}

      void write(std::ostream& os) const;

      ProcessHistoryID const& processHistoryID() const { return extra_.processHistoryID_; }
      EventID const& id() const { return id_; }
      std::string const& processGUID() const { return processGUID_; }
      Timestamp const& time() const { return time_; }
      LuminosityBlockNumber_t luminosityBlock() const { return id_.luminosityBlock(); }
      EventNumber_t event() const { return id_.event(); }
      RunNumber_t run() const { return id_.run(); }
      bool isRealData() const { return extra_.isRealData_; }
      ExperimentType experimentType() const { return extra_.experimentType_; }
      int bunchCrossing() const { return bunchCrossing_; }
      int orbitNumber() const { return orbitNumber_; }
      int storeNumber() const { return extra_.storeNumber_; }

      EventAuxiliary eventAuxiliary() const {
        auto ea{EventAuxiliary(id_,
                               processGUID_,
                               time_,
                               extra_.isRealData_,
                               extra_.experimentType_,
                               bunchCrossing_,
                               extra_.storeNumber_,
                               orbitNumber_)};
        ea.setProcessHistoryID(extra_.processHistoryID_);
        return ea;
      }

    private:
      template <typename T, typename C>
      const T& memoize(const T& item, C& memopad) const {
        auto it = memopad.insert(item);
        return *it.first;
      }

      // Event ID
      EventID id_;
      // Globally unique process ID of process that created event.
      const std::string& processGUID_;
      // Time from DAQ
      Timestamp time_;
      //  The bunch crossing number
      int bunchCrossing_;
      // The orbit number
      int orbitNumber_;
      // the stuff that changes slowly
      const CompactEventAuxiliaryExtra& extra_;
    };

    using value_type = CompactEventAuxiliary;
    using iterator = std::vector<value_type>::iterator;
    using size_type = std::vector<value_type>::size_type;
    using const_iterator = std::vector<value_type>::const_iterator;

    size_type size() const { return compactAuxiliaries_.size(); }
    void reserve(std::size_t size) { compactAuxiliaries_.reserve(size); }
    const_iterator begin() const { return compactAuxiliaries_.begin(); }
    const_iterator end() const { return compactAuxiliaries_.end(); }
    const_iterator cbegin() const { return compactAuxiliaries_.cbegin(); }
    const_iterator cend() const { return compactAuxiliaries_.cend(); }

    size_type extrasSize() const { return extras_.size(); }
    size_type guidsSize() const { return processGUIDs_.size(); }

    void push_back(const EventAuxiliary& ea) { compactAuxiliaries_.emplace_back(ea, processGUIDs_, extras_); }

  private:
    // Items that change every event
    std::vector<CompactEventAuxiliary> compactAuxiliaries_;
    // Items that change relatively infrequently
    extraMemo extras_;
    // Globally unique process IDs of processes that created events.
    GUIDmemo processGUIDs_;
  };

  inline std::ostream& operator<<(std::ostream& os, const CompactEventAuxiliaryVector::CompactEventAuxiliary& p) {
    p.write(os);
    return os;
  }

  inline std::ostream& operator<<(std::ostream& os, const CompactEventAuxiliaryVector::CompactEventAuxiliaryExtra& p) {
    p.write(os);
    return os;
  }
}  // namespace edm

#endif
