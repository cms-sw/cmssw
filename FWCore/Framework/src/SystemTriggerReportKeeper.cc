// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SystemTriggerReportKeeper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon, 07 Jul 2014 14:37:32 GMT
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "FWCore/Framework/src/TriggerReport.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/SystemTriggerReportKeeper.h"

using namespace edm;

namespace {
  bool lessModuleDescription(const ModuleDescription* iLHS, const ModuleDescription* iRHS) {
    return iLHS->id() < iRHS->id();
  }
}  // namespace
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SystemTriggerReportKeeper::SystemTriggerReportKeeper(unsigned int iNumStreams,
                                                     std::vector<const ModuleDescription*> const& iModules,
                                                     service::TriggerNamesService const& iNamesService,
                                                     ProcessContext const* iProcessContext)
    : m_streamPathStatus(iNumStreams), m_modules(iModules), m_processContext(iProcessContext), m_minModuleID(0) {
  std::sort(m_modules.begin(), m_modules.end(), lessModuleDescription);
  if (not m_modules.empty()) {
    m_minModuleID = m_modules.front()->id();
    unsigned int numModuleSlots = m_modules.back()->id() - m_minModuleID + 1;
    m_streamModuleStatus.resize(iNumStreams);
    for (auto& stream : m_streamModuleStatus) {
      stream.resize(numModuleSlots);
    }
  }

  std::vector<unsigned int> numModulesInPath;
  std::vector<unsigned int> numModulesInEndPath;

  const unsigned int numPaths = iNamesService.getTrigPaths().size();
  const unsigned int numEndPaths = iNamesService.getEndPaths().size();
  m_pathNames.reserve(numPaths + numEndPaths);
  std::copy(iNamesService.getTrigPaths().begin(), iNamesService.getTrigPaths().end(), std::back_inserter(m_pathNames));
  std::copy(iNamesService.getEndPaths().begin(), iNamesService.getEndPaths().end(), std::back_inserter(m_pathNames));

  numModulesInPath.reserve(numPaths);
  numModulesInEndPath.reserve(numEndPaths);

  m_modulesOnPaths.reserve(numPaths + numEndPaths);

  for (unsigned int i = 0; i < numPaths; ++i) {
    numModulesInPath.push_back(iNamesService.getTrigPathModules(i).size());
    m_modulesOnPaths.push_back(iNamesService.getTrigPathModules(i));
  }
  for (unsigned int i = 0; i < numEndPaths; ++i) {
    numModulesInEndPath.push_back(iNamesService.getEndPathModules(i).size());
    m_modulesOnPaths.push_back(iNamesService.getEndPathModules(i));
  }

  m_endPathOffset = numModulesInPath.size();

  for (auto& stream : m_streamPathStatus) {
    unsigned int index = 0;
    stream.resize(numModulesInPath.size() + numModulesInEndPath.size());
    for (unsigned int numMods : numModulesInPath) {
      stream[index].m_moduleStatus.resize(numMods);
      ++index;
    }
    for (unsigned int numMods : numModulesInEndPath) {
      stream[index].m_moduleStatus.resize(numMods);
      ++index;
    }
  }
}

//
// member functions
//
void SystemTriggerReportKeeper::removeModuleIfExists(ModuleDescription const& module) {
  auto found = std::lower_bound(m_modules.begin(), m_modules.end(), &module, lessModuleDescription);
  if (*found == &module) {
    m_modules.erase(found);
  }
}

SystemTriggerReportKeeper::PathStatus& SystemTriggerReportKeeper::pathStatus(StreamContext const& iStream,
                                                                             PathContext const& iPath) {
  unsigned int offset = 0;
  if (iPath.isEndPath()) {
    offset = m_endPathOffset;
  }
  assert(iPath.pathID() + offset < m_streamPathStatus[iStream.streamID().value()].size());
  return m_streamPathStatus[iStream.streamID().value()][iPath.pathID() + offset];
}

//NOTE: Have to check bounds rather than ProcessContext on the
// module callbacks.
inline bool SystemTriggerReportKeeper::checkBounds(unsigned int id) const {
  return id >= m_minModuleID and id < m_streamModuleStatus.front().size() + m_minModuleID;
}

void SystemTriggerReportKeeper::stopPath(StreamContext const& iStream,
                                         PathContext const& iPath,
                                         HLTPathStatus const& iStatus) {
  if (m_processContext == iStream.processContext()) {
    auto& pStatus = pathStatus(iStream, iPath);

    if (iStatus.accept()) {
      ++pStatus.m_timesPassed;
    } else if (iStatus.error()) {
      ++pStatus.m_timesExcept;
    } else if (iStatus.wasrun()) {
      ++pStatus.m_timesFailed;
    }
    ++pStatus.m_timesRun;

    //mark all modules up to and including the decision module as being visited
    auto& modsOnPath = pStatus.m_moduleStatus;
    assert(iStatus.index() < modsOnPath.size());
    for (unsigned int i = 0; i < iStatus.index(); ++i) {
      ++modsOnPath[i].m_timesVisited;
      ++modsOnPath[i].m_timesPassed;
    }
    ++modsOnPath[iStatus.index()].m_timesVisited;
    if (iStatus.error()) {
      ++modsOnPath[iStatus.index()].m_timesExcept;
    } else if (iStatus.accept()) {
      ++modsOnPath[iStatus.index()].m_timesPassed;
    } else if (iStatus.wasrun()) {
      ++modsOnPath[iStatus.index()].m_timesFailed;
    }
  }
}

void SystemTriggerReportKeeper::stopModuleEvent(StreamContext const& iStream, ModuleCallingContext const& iModule) {
  if (checkBounds(iModule.moduleDescription()->id())) {
    auto& mod = m_streamModuleStatus[iStream.streamID().value()][iModule.moduleDescription()->id() - m_minModuleID];

    if (iModule.state() == ModuleCallingContext::State::kFinishedPassed) {
      ++(mod.m_timesPassed);
    } else if (iModule.state() == ModuleCallingContext::State::kFinishedFailed) {
      ++(mod.m_timesFailed);
    } else if (iModule.state() == ModuleCallingContext::State::kException) {
      ++(mod.m_timesExcept);
    }
    ++(mod.m_timesRun);
  }
}

void SystemTriggerReportKeeper::checkModuleAcquire(StreamContext const& iStream, ModuleCallingContext const& iModule) {
  if (iModule.state() == ModuleCallingContext::State::kException and checkBounds(iModule.moduleDescription()->id())) {
    auto& mod = m_streamModuleStatus[iStream.streamID().value()][iModule.moduleDescription()->id() - m_minModuleID];
    ++(mod.m_timesRun);
    ++(mod.m_timesExcept);
  }
}

static void fillPathSummary(unsigned int iStartIndex,
                            unsigned int iEndIndex,
                            std::vector<std::string> const& iPathNames,
                            std::vector<std::vector<std::string>> const& iModulesOnPaths,
                            std::vector<std::vector<SystemTriggerReportKeeper::PathStatus>> const& iPathStatuses,
                            std::vector<PathSummary>& iSummary) {
  iSummary.resize(iEndIndex - iStartIndex);
  for (auto const& stream : iPathStatuses) {
    auto it = iSummary.begin();
    for (unsigned int index = iStartIndex; index < iEndIndex; ++index, ++it) {
      assert(it != iSummary.end());
      assert(index < stream.size());
      auto const& pathStatus = stream[index];
      assert(index < iPathNames.size());
      it->name = iPathNames[index];
      it->bitPosition = index - iStartIndex;
      if (not pathStatus.m_moduleStatus.empty()) {
        it->timesRun += pathStatus.m_timesRun;
        it->timesPassed += pathStatus.m_timesPassed;
        it->timesFailed += pathStatus.m_timesFailed;
        it->timesExcept += pathStatus.m_timesExcept;
      }
      if (it->moduleInPathSummaries.empty()) {
        it->moduleInPathSummaries.resize(pathStatus.m_moduleStatus.size());
      }
      for (unsigned int modIndex = 0; modIndex < pathStatus.m_moduleStatus.size(); ++modIndex) {
        assert(modIndex < pathStatus.m_moduleStatus.size());
        auto const& modStatus = pathStatus.m_moduleStatus[modIndex];
        assert(modIndex < it->moduleInPathSummaries.size());
        auto& modSummary = it->moduleInPathSummaries[modIndex];
        modSummary.bitPosition = modIndex;
        if (modSummary.moduleLabel.empty()) {
          assert(index < iModulesOnPaths.size());
          assert(modIndex < iModulesOnPaths[index].size());
          auto modLabel = iModulesOnPaths[index][modIndex];
          modSummary.moduleLabel = modLabel;
        }
        modSummary.timesVisited += modStatus.m_timesVisited;
        modSummary.timesPassed += modStatus.m_timesPassed;
        modSummary.timesFailed += modStatus.m_timesFailed;
        modSummary.timesExcept += modStatus.m_timesExcept;
      }
    }
  }
}

void SystemTriggerReportKeeper::fillTriggerReport(TriggerReport& rep) const {
  //Per module summary
  {
    auto& summary = rep.workerSummaries;
    summary.resize(m_modules.size());
    //Figure out how often a module was visited
    std::unordered_map<std::string, unsigned int> visited;
    for (auto const& stream : m_streamPathStatus) {
      unsigned int pathIndex = 0;
      for (auto const& path : stream) {
        unsigned int modIndex = 0;
        for (auto const& mod : path.m_moduleStatus) {
          visited[m_modulesOnPaths[pathIndex][modIndex]] += mod.m_timesVisited;
          ++modIndex;
        }
        ++pathIndex;
      }
    }

    unsigned int modIndex = 0;
    for (auto const& mod : m_modules) {
      auto& outMod = summary[modIndex];
      outMod.moduleLabel = mod->moduleLabel();
      outMod.timesVisited = 0;
      outMod.timesRun = 0;
      outMod.timesPassed = 0;
      outMod.timesFailed = 0;
      outMod.timesExcept = 0;
      auto moduleId = mod->id() - m_minModuleID;
      for (auto const& stream : m_streamModuleStatus) {
        assert(moduleId < stream.size());
        auto const& status = stream[moduleId];
        outMod.timesRun += status.m_timesRun;
        outMod.timesPassed += status.m_timesPassed;
        outMod.timesFailed += status.m_timesFailed;
        outMod.timesExcept += status.m_timesExcept;
      }
      outMod.timesVisited = visited[mod->moduleLabel()];
      if (0 == outMod.timesVisited) {
        outMod.timesVisited = outMod.timesRun;
      }
      ++modIndex;
    }
  }
  sort_all(rep.workerSummaries);

  //Per path summary
  {
    fillPathSummary(0, m_endPathOffset, m_pathNames, m_modulesOnPaths, m_streamPathStatus, rep.trigPathSummaries);
    fillPathSummary(m_endPathOffset,
                    m_streamPathStatus[0].size(),
                    m_pathNames,
                    m_modulesOnPaths,
                    m_streamPathStatus,
                    rep.endPathSummaries);
  }
}

//
// const member functions
//

//
// static member functions
//
