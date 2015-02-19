// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SystemTimeKeeper
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
#include "FWCore/Framework/interface/TriggerTimingReport.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "SystemTimeKeeper.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SystemTimeKeeper::SystemTimeKeeper(unsigned int iNumStreams,
                                   std::vector<const ModuleDescription*> const& iModules,
                                   service::TriggerNamesService const& iNamesService):
m_streamEventTimer(iNumStreams),
m_streamPathTiming(iNumStreams),
m_modules(iModules),
m_minModuleID(0),
m_numberOfEvents(0)
{
  std::sort(m_modules.begin(),m_modules.end(),
            [] (const ModuleDescription* iLHS,
                      const ModuleDescription* iRHS) -> bool {
              return iLHS->id() < iRHS->id();
            });
  if(not m_modules.empty()) {
    m_minModuleID = m_modules.front()->id();
    unsigned int numModuleSlots = m_modules.back()->id() - m_minModuleID + 1;
    m_streamModuleTiming.resize(iNumStreams);
    for(auto& stream: m_streamModuleTiming) {
      stream.resize(numModuleSlots);
    }
  }
  
  
  std::vector<unsigned int> numModulesInPath;
  std::vector<unsigned int> numModulesInEndPath;
  
  const unsigned int numPaths = iNamesService.getTrigPaths().size();
  const unsigned int numEndPaths = iNamesService.getEndPaths().size();
  m_pathNames.reserve(numPaths+numEndPaths);
  std::copy(iNamesService.getTrigPaths().begin(),
            iNamesService.getTrigPaths().end(),
            std::back_inserter(m_pathNames));
  std::copy(iNamesService.getEndPaths().begin(),
            iNamesService.getEndPaths().end(),
            std::back_inserter(m_pathNames));
  
  numModulesInPath.reserve(numPaths);
  numModulesInEndPath.reserve(numEndPaths);
  
  m_modulesOnPaths.reserve(numPaths+numEndPaths);
  
  for(unsigned int i =0; i<numPaths;++i) {
    numModulesInPath.push_back(iNamesService.getTrigPathModules(i).size());
    m_modulesOnPaths.push_back(iNamesService.getTrigPathModules(i));
  }
  for(unsigned int i =0; i<numEndPaths;++i) {
    numModulesInEndPath.push_back(iNamesService.getEndPathModules(i).size());
    m_modulesOnPaths.push_back(iNamesService.getEndPathModules(i));
  }
  
  m_endPathOffset =numModulesInPath.size();

  for( auto& stream: m_streamPathTiming) {
    unsigned int index = 0;
    stream.resize(numModulesInPath.size()+numModulesInEndPath.size());
    for(unsigned int numMods : numModulesInPath) {
      stream[index].m_moduleTiming.resize(numMods);
      ++index;
    }
    for(unsigned int numMods : numModulesInEndPath) {
      stream[index].m_moduleTiming.resize(numMods);
      ++index;
    }
    
  }
}

//
// member functions
//
SystemTimeKeeper::PathTiming&
SystemTimeKeeper::pathTiming(StreamContext const& iStream,
                             PathContext const& iPath) {
  unsigned int offset = 0;
  if(iPath.isEndPath()) {
    offset = m_endPathOffset;
  }
  assert(iPath.pathID()+offset < m_streamPathTiming[iStream.streamID().value()].size());
  return m_streamPathTiming[iStream.streamID().value()][iPath.pathID()+offset];
}



void
SystemTimeKeeper::startEvent(StreamID iID) {
  m_numberOfEvents++;
  m_streamEventTimer[iID.value()].start();
}

void
SystemTimeKeeper::stopEvent(StreamContext const& iContext) {
  m_streamEventTimer[iContext.streamID().value()].stop();
}

void
SystemTimeKeeper::startPath(StreamContext const& iStream,
                            PathContext const& iPath) {
  auto& timing = pathTiming(iStream,iPath);
  timing.m_timer.start();
}

void
SystemTimeKeeper::stopPath(StreamContext const& iStream,
                           PathContext const& iPath,
                           HLTPathStatus const& iStatus) {
  auto& timing = pathTiming(iStream,iPath);
  timing.m_timer.stop();
  
  //mark all modules up to and including the decision module as being visited
  auto& modsOnPath = timing.m_moduleTiming;
  for(unsigned int i = 0; i< iStatus.index()+1;++i) {
    ++modsOnPath[i].m_timesVisited;
  }
}


void
SystemTimeKeeper::startModuleEvent(StreamContext const& iStream, ModuleCallingContext const& iModule) {
  auto& mod =
  m_streamModuleTiming[iStream.streamID().value()][iModule.moduleDescription()->id()-m_minModuleID];
  mod.m_timer.start();
  ++(mod.m_timesRun);
  
}
void SystemTimeKeeper::stopModuleEvent(StreamContext const& iStream,
                                       ModuleCallingContext const& iModule) {
  auto& mod =
  m_streamModuleTiming[iStream.streamID().value()][iModule.moduleDescription()->id()-m_minModuleID];
  auto times = mod.m_timer.stop();
  
  if(iModule.type() == ParentContext::Type::kPlaceInPath ) {
    auto place = iModule.placeInPathContext();
    
    auto& modTiming = pathTiming(iStream,*(place->pathContext())).m_moduleTiming[place->placeInPath()];
    modTiming.m_realTime += times;
  }

}
void SystemTimeKeeper::pauseModuleEvent(StreamContext const& iStream,
                                        ModuleCallingContext const& iModule) {
  auto& mod =
  m_streamModuleTiming[iStream.streamID().value()][iModule.moduleDescription()->id()-m_minModuleID];
  auto times = mod.m_timer.stop();
  
  if(iModule.type() == ParentContext::Type::kPlaceInPath ) {
    auto place = iModule.placeInPathContext();
    
    auto& modTiming = pathTiming(iStream,*(place->pathContext())).m_moduleTiming[place->placeInPath()];
    modTiming.m_realTime += times;
  }

}
void
SystemTimeKeeper::restartModuleEvent(StreamContext const& iStream,
                                     ModuleCallingContext const& iModule) {
  auto& mod =
  m_streamModuleTiming[iStream.streamID().value()][iModule.moduleDescription()->id()-m_minModuleID];
  mod.m_timer.start();
}

void
SystemTimeKeeper::startProcessingLoop() {
  m_processingLoopTimer.start();
}

void
SystemTimeKeeper::stopProcessingLoop() {
  m_processingLoopTimer.stop();
}


static void
fillPathSummary(unsigned int iStartIndex,
                unsigned int iEndIndex,
                std::vector<std::string> const& iPathNames,
                std::vector<std::vector<std::string>> const& iModulesOnPaths,
                std::vector<std::vector<SystemTimeKeeper::PathTiming>> const& iPathTimings,
                std::vector<PathTimingSummary>& iSummary) {
  iSummary.resize(iEndIndex-iStartIndex);

  for(auto const& stream: iPathTimings) {
    auto it = iSummary.begin();
    for(unsigned int index = iStartIndex; index < iEndIndex; ++index, ++it) {
      auto const& pathTiming = stream[index];
      it->name = iPathNames[index];
      it->bitPosition = index-iStartIndex;
      if(not pathTiming.m_moduleTiming.empty()) {
        it->timesRun += pathTiming.m_moduleTiming[0].m_timesVisited;
      }
      it->realTime += pathTiming.m_timer.realTime();
      if(it->moduleInPathSummaries.empty()) {
        it->moduleInPathSummaries.resize(pathTiming.m_moduleTiming.size());
      }
      for(unsigned int modIndex=0; modIndex < pathTiming.m_moduleTiming.size(); ++modIndex) {
        auto const& modTiming =pathTiming.m_moduleTiming[modIndex];
        auto& modSummary =it->moduleInPathSummaries[modIndex];
        if(modSummary.moduleLabel.empty()) {
          modSummary.moduleLabel = iModulesOnPaths[index][modIndex];
        }
        modSummary.timesVisited += modTiming.m_timesVisited;
        modSummary.realTime += modTiming.m_realTime;
      }
    }
  }
}

void
SystemTimeKeeper::fillTriggerTimingReport( TriggerTimingReport& rep) {
  {
    rep.eventSummary.totalEvents = m_numberOfEvents;
    double sumEventTime = 0.;
    for(auto const& stream: m_streamEventTimer) {
      sumEventTime += stream.realTime();
    }
    rep.eventSummary.realTime = m_processingLoopTimer.realTime();
    rep.eventSummary.cpuTime = m_processingLoopTimer.cpuTime();
    rep.eventSummary.sumStreamRealTime = sumEventTime;
  }
  
  //Per module summary
  {
    auto& summary = rep.workerSummaries;
    summary.resize(m_modules.size());
    //Figure out how often a module was visited
    std::map<std::string,unsigned int> visited;
    for(auto const& stream: m_streamPathTiming) {
      unsigned int pathIndex = 0;
      for(auto const& path: stream) {
        unsigned int modIndex = 0;
        for(auto const& mod: path.m_moduleTiming) {
          visited[m_modulesOnPaths[pathIndex][modIndex]] += mod.m_timesVisited;
          ++modIndex;
        }
        ++pathIndex;
      }
    }

    unsigned int modIndex=0;
    for(auto const& mod: m_modules) {
      auto& outMod = summary[modIndex];
      outMod.moduleLabel = mod->moduleLabel();
      outMod.realTime = 0.;
      
      auto moduleId =mod->id()-m_minModuleID;
      for(auto const& stream: m_streamModuleTiming) {
        auto const& timing = stream[moduleId];
        outMod.realTime += timing.m_timer.realTime();
        outMod.timesRun += timing.m_timesRun;
      }
      outMod.timesVisited = visited[mod->moduleLabel()];
      if(0 == outMod.timesVisited) {
        outMod.timesVisited = outMod.timesRun;
      }
      ++modIndex;
    }
  }
  
  //Per path summary
  {
    fillPathSummary(0, m_endPathOffset, m_pathNames, m_modulesOnPaths, m_streamPathTiming, rep.trigPathSummaries);
    fillPathSummary(m_endPathOffset, m_streamPathTiming[0].size(), m_pathNames, m_modulesOnPaths, m_streamPathTiming, rep.endPathSummaries);
  }
}


//
// const member functions
//

//
// static member functions
//
