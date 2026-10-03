#ifndef FWCore_Framework_UnscheduledConfigurator_h
#define FWCore_Framework_UnscheduledConfigurator_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     UnscheduledConfigurator
//
/**\class UnscheduledConfigurator UnscheduledConfigurator.h "UnscheduledConfigurator.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 13 Apr 2016 18:57:55 GMT
//

// system include files
#include <unordered_map>
#include <variant>

// user include files

// forward declarations

namespace edm {
  template <typename TI, typename TP>
  class TransitionWorker;

  class EventTransitionInfo;
  class RunTransitionInfo;
  class LumiTransitionInfo;
  class ProcessBlockTransitionInfo;
  class InputProcessBlockTransitionInfo;
  struct TransitionPhaseGlobal;
  class UnscheduledAuxiliary;

  class UnscheduledConfigurator {
  public:
    using GlobalWorkerTypePtr = std::variant<std::monostate,
                                             TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>*,
                                             TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>*,
                                             TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>*,
                                             TransitionWorker<ProcessBlockTransitionInfo, TransitionPhaseGlobal>*,
                                             TransitionWorker<InputProcessBlockTransitionInfo, TransitionPhaseGlobal>*>;
    template <typename IT>
    UnscheduledConfigurator(IT iBegin, IT iEnd, UnscheduledAuxiliary const* iAux) : m_aux(iAux) {
      for (auto it = iBegin; it != iEnd; ++it) {
        m_labelToWorker.emplace((*it)->description()->moduleLabel(), *it);
      }
    }

    UnscheduledConfigurator(const UnscheduledConfigurator&) = delete;                   // stop default
    const UnscheduledConfigurator& operator=(const UnscheduledConfigurator&) = delete;  // stop default

    // ---------- const member functions ---------------------
    GlobalWorkerTypePtr findWorker(std::string const& iLabel) const {
      auto itFound = m_labelToWorker.find(iLabel);
      if (itFound != m_labelToWorker.end()) {
        return itFound->second;
      }
      return GlobalWorkerTypePtr();
    }

    UnscheduledAuxiliary const* auxiliary() const { return m_aux; }

  private:
    // ---------- member data --------------------------------
    std::unordered_map<std::string, GlobalWorkerTypePtr> m_labelToWorker;
    UnscheduledAuxiliary const* m_aux;
  };
}  // namespace edm

#endif
