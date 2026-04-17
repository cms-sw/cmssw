#ifndef FWCore_Framework_SourceStatus_h
#define FWCore_Framework_SourceStatus_h
// Package:     FWCore/Framework
// Class  :     SourceStatus
/// Description: Keep status information about the source
//
#include "FWCore/Framework/interface/InputSource.h"

#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"

#include <variant>
namespace edm {
  class SourceStatus {
  public:
    SourceStatus() = default;
    explicit SourceStatus(InputSource::ItemTypeInfo const& iType) noexcept : nextTransitionType_(iType) {}
    explicit SourceStatus(InputSource::ItemType iType) noexcept : nextTransitionType_(iType) {}
    SourceStatus(SourceStatus const&) noexcept = default;
    SourceStatus(SourceStatus&&) noexcept = default;
    SourceStatus& operator=(SourceStatus const&) noexcept = default;
    SourceStatus& operator=(SourceStatus&&) noexcept = default;

    InputSource::ItemTypeInfo nextTransitionType() const noexcept { return nextTransitionType_; }
    void setNextTransitionType(InputSource::ItemTypeInfo const& iType) noexcept { nextTransitionType_ = iType; }
    void setNextTransitionType(InputSource::ItemType iType) noexcept {
      nextTransitionType_ = InputSource::ItemTypeInfo{iType};
    }

    bool needToCallNext() const noexcept { return needToCallNext_; }
    void setNeedToCallNext(bool val) noexcept { needToCallNext_ = val; }

    edm::RunAuxiliary const* runAuxiliary() const noexcept {
      if (auto p = std::get_if<RunAuxiliary>(&currentRunOrLumiAux_)) {
        return p;
      }
      return nullptr;
    }
    edm::LuminosityBlockAuxiliary const* lumiAuxiliary() const noexcept {
      if (auto p = std::get_if<LuminosityBlockAuxiliary>(&currentRunOrLumiAux_)) {
        return p;
      }
      return nullptr;
    }
    edm::ProcessHistoryID const& reducedProcessHistoryID() const noexcept {
      if (reducedProcessHistoryID_) {
        return *reducedProcessHistoryID_;
      }
      static const edm::ProcessHistoryID emptyID;
      return emptyID;
    }

    void setRunAuxiliary(edm::RunAuxiliary const& aux) { currentRunOrLumiAux_ = aux; }
    void setLuminosityBlockAuxiliary(edm::LuminosityBlockAuxiliary const& aux) { currentRunOrLumiAux_ = aux; }
    void setReducedProcessHistoryID(edm::ProcessHistoryID const& id) { reducedProcessHistoryID_ = id; }

  private:
    InputSource::ItemTypeInfo nextTransitionType_{InputSource::ItemType::IsInvalid};
    std::variant<std::monostate, RunAuxiliary, LuminosityBlockAuxiliary> currentRunOrLumiAux_{};
    std::optional<edm::ProcessHistoryID> reducedProcessHistoryID_;
    bool needToCallNext_ = true;
  };
}  // namespace edm
#endif
