// -*- C++ -*-
//
// Package:     Services
// Class  :     CheckTransitions
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Sep  8 14:17:58 EDT 2005
//
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <vector>
#include <string>
#include "tbb/concurrent_vector.h"
#include <iostream>

namespace edm {

  namespace service {
    class CheckTransitions {
    public:
      enum class Phase { kBeginRun, kBeginLumi, kEvent, kEndLumi, kEndRun };

      enum class Transition { IsInvalid, IsStop, IsFile, IsRun, IsLumi, IsEvent };

      CheckTransitions(const ParameterSet&, ActivityRegistry&);
      ~CheckTransitions() noexcept(false);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void preallocate(service::SystemBounds const&);

      void preBeginJob(PathsAndConsumesOfModulesBase const&, ProcessContext const&);
      void postEndJob();

      void preOpenFile(std::string const&, bool);

      void preCloseFile(std::string const& lfn, bool primary);

      void preGlobalBeginRun(GlobalContext const&);

      void preGlobalEndRun(GlobalContext const&);

      void preStreamBeginRun(StreamContext const&);

      void preStreamEndRun(StreamContext const&);

      void preGlobalBeginLumi(GlobalContext const&);

      void preGlobalEndLumi(GlobalContext const&);

      void preStreamBeginLumi(StreamContext const&);

      void preStreamEndLumi(StreamContext const&);

      void preEvent(StreamContext const&);

    private:
      tbb::concurrent_vector<std::tuple<Phase, edm::EventID, int>> m_seenTransitions;
      std::vector<std::pair<Transition, edm::EventID>> m_expectedTransitions;
      int m_nstreams = 0;
      bool m_failed = false;
    };
  }  // namespace service
}  // namespace edm
using namespace edm::service;

namespace {
  using Phase = CheckTransitions::Phase;
  using Transition = CheckTransitions::Transition;

  Transition stringToType(const std::string& iTrans) {
    if (iTrans == "IsStop") {
      return Transition::IsStop;
    }
    if (iTrans == "IsFile") {
      return Transition::IsFile;
    }
    if (iTrans == "IsRun") {
      return Transition::IsRun;
    }
    if (iTrans == "IsLumi") {
      return Transition::IsLumi;
    }
    if (iTrans == "IsEvent") {
      return Transition::IsEvent;
    }

    throw edm::Exception(edm::errors::Configuration) << "Unknown transition type \'" << iTrans << "\'";

    return Transition::IsInvalid;
  }

  std::vector<std::tuple<Phase, edm::EventID, int>> expectedValues(
      std::vector<std::pair<Transition, edm::EventID>> const& iTrans, int iNStreams) {
    std::vector<std::tuple<Phase, edm::EventID, int>> returnValue;
    returnValue.reserve(iTrans.size());

    const edm::RunNumber_t maxIDValue = edm::EventID::maxRunNumber();
    edm::EventID lastRun = {maxIDValue, 0, 0};
    edm::EventID lastLumi = {maxIDValue, maxIDValue, 0};
    for (auto const& tran : iTrans) {
      switch (tran.first) {
        case Transition::IsFile: {
          break;
        }
        case Transition::IsRun: {
          if (tran.second != lastRun) {
            if (lastRun.run() != maxIDValue) {
              //end transitions
              for (int i = 0; i < iNStreams; ++i) {
                returnValue.emplace_back(Phase::kEndRun, lastRun, i);
              }
              returnValue.emplace_back(Phase::kEndRun, lastRun, 1000);
            }
            //begin transitions
            returnValue.emplace_back(Phase::kBeginRun, tran.second, -1);
            for (int i = 0; i < iNStreams; ++i) {
              returnValue.emplace_back(Phase::kBeginRun, tran.second, i);
            }
            lastRun = tran.second;
          }
          break;
        }
        case Transition::IsLumi: {
          if (tran.second != lastLumi) {
            if (lastLumi.run() != maxIDValue) {
              //end transitions
              for (int i = 0; i < iNStreams; ++i) {
                returnValue.emplace_back(Phase::kEndLumi, lastLumi, i);
              }
              returnValue.emplace_back(Phase::kEndLumi, lastLumi, 1000);
            }
            //begin transitions
            returnValue.emplace_back(Phase::kBeginLumi, tran.second, -1);
            for (int i = 0; i < iNStreams; ++i) {
              returnValue.emplace_back(Phase::kBeginLumi, tran.second, i);
            }
            lastLumi = tran.second;
          }
          break;
        }
        case Transition::IsEvent: {
          returnValue.emplace_back(Phase::kEvent, tran.second, -2);
        }
        case Transition::IsStop:
        case Transition::IsInvalid: {
          break;
        }
      }
    }
    if (lastLumi.run() != maxIDValue) {
      //end transitions
      for (int i = 0; i < iNStreams; ++i) {
        returnValue.emplace_back(Phase::kEndLumi, lastLumi, i);
      }
      returnValue.emplace_back(Phase::kEndLumi, lastLumi, 1000);
    }
    if (lastRun.run() != maxIDValue) {
      //end transitions
      for (int i = 0; i < iNStreams; ++i) {
        returnValue.emplace_back(Phase::kEndRun, lastRun, i);
      }
      returnValue.emplace_back(Phase::kEndRun, lastRun, 1000);
    }
    return returnValue;
  }

}  // namespace

CheckTransitions::CheckTransitions(ParameterSet const& iPS, ActivityRegistry& iRegistry) {
  for (auto const& p : iPS.getUntrackedParameter<std::vector<edm::ParameterSet>>("transitions")) {
    m_expectedTransitions.emplace_back(stringToType(p.getUntrackedParameter<std::string>("type")),
                                       p.getUntrackedParameter<EventID>("id"));
  }

  iRegistry.watchPreallocate(this, &CheckTransitions::preallocate);

  iRegistry.watchPostEndJob(this, &CheckTransitions::postEndJob);

  iRegistry.watchPreOpenFile(this, &CheckTransitions::preOpenFile);

  iRegistry.watchPreCloseFile(this, &CheckTransitions::preCloseFile);

  iRegistry.watchPreGlobalBeginRun(this, &CheckTransitions::preGlobalBeginRun);

  iRegistry.watchPreGlobalEndRun(this, &CheckTransitions::preGlobalEndRun);

  iRegistry.watchPreStreamBeginRun(this, &CheckTransitions::preStreamBeginRun);

  iRegistry.watchPreStreamEndRun(this, &CheckTransitions::preStreamEndRun);

  iRegistry.watchPreGlobalBeginLumi(this, &CheckTransitions::preGlobalBeginLumi);

  iRegistry.watchPreGlobalEndLumi(this, &CheckTransitions::preGlobalEndLumi);

  iRegistry.watchPreStreamBeginLumi(this, &CheckTransitions::preStreamBeginLumi);

  iRegistry.watchPreStreamEndLumi(this, &CheckTransitions::preStreamEndLumi);

  iRegistry.watchPreEvent(this, &CheckTransitions::preEvent);
}

CheckTransitions::~CheckTransitions() noexcept(false) {
  if (m_failed) {
    throw edm::Exception(errors::EventProcessorFailure) << "incorrect transtions";
  }
}

void CheckTransitions::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.setComment("Checks that the transitions specified occur during the job.");

  ParameterSetDescription trans;
  trans.addUntracked<std::string>("type");
  trans.addUntracked<edm::EventID>("id");
  desc.addVPSetUntracked("transitions", trans, {{}});
  descriptions.add("CheckTransitions", desc);
}

void CheckTransitions::preallocate(service::SystemBounds const& bounds) { m_nstreams = bounds.maxNumberOfStreams(); }

void CheckTransitions::postEndJob() {
  auto expectedV = expectedValues(m_expectedTransitions, m_nstreams);

  std::vector<std::tuple<Phase, edm::EventID, int>> orderedSeen;
  orderedSeen.reserve(m_seenTransitions.size());
  for (auto const& i : m_seenTransitions) {
    //      std::cout <<i.first.m_run<<" "<<i.first.m_lumi<<" "<<i.first.m_event<<" "<<i.second<<std::endl;
    auto s = std::get<2>(i);
    if (std::get<1>(i).event() > 0) {
      s = -2;
    }
    orderedSeen.emplace_back(std::get<0>(i), std::get<1>(i), s);
  }
  std::sort(orderedSeen.begin(), orderedSeen.end());

  auto orderedExpected = expectedV;
  std::sort(orderedExpected.begin(), orderedExpected.end());
  /*   for(auto const& i: expectedV) {
   std::cout <<i.first.m_run<<" "<<i.first.m_lumi<<" "<<i.first.m_event<<" "<<i.second<<std::endl;
   } */

  auto itOS = orderedSeen.begin();
  for (auto itOE = orderedExpected.begin(); itOE != orderedExpected.end(); ++itOE) {
    if (itOS == orderedSeen.end()) {
      break;
    }
    if (*itOE != *itOS) {
      auto syncOE = std::get<1>(*itOE);
      auto syncOS = std::get<1>(*itOS);
      std::cout << "Different ordering " << syncOE << " " << std::get<2>(*itOE) << "\n"
                << "                   " << syncOS << " " << std::get<2>(*itOS) << "\n";
      m_failed = true;
    }
    ++itOS;
  }

  if (orderedSeen.size() != orderedExpected.size()) {
    std::cout << "Wrong number of transition " << orderedSeen.size() << " " << orderedExpected.size() << std::endl;
    m_failed = true;
    return;
  }
}

void CheckTransitions::preOpenFile(std::string const& lfn, bool b) {}

void CheckTransitions::preCloseFile(std::string const& lfn, bool b) {}

void CheckTransitions::preGlobalBeginRun(GlobalContext const& gc) {
  auto id = gc.luminosityBlockID();
  m_seenTransitions.emplace_back(Phase::kBeginRun, edm::EventID{id.run(), 0, 0}, -1);
}

void CheckTransitions::preGlobalEndRun(GlobalContext const& gc) {
  auto id = gc.luminosityBlockID();
  m_seenTransitions.emplace_back(Phase::kEndRun, edm::EventID{id.run(), 0, 0}, 1000);
}

void CheckTransitions::preStreamBeginRun(StreamContext const& sc) {
  m_seenTransitions.emplace_back(Phase::kBeginRun, sc.eventID(), sc.streamID());
}

void CheckTransitions::preStreamEndRun(StreamContext const& sc) {
  m_seenTransitions.emplace_back(Phase::kEndRun, sc.eventID(), sc.streamID());
}

void CheckTransitions::preGlobalBeginLumi(GlobalContext const& gc) {
  auto id = gc.luminosityBlockID();
  m_seenTransitions.emplace_back(Phase::kBeginLumi, edm::EventID{id.run(), id.luminosityBlock(), 0}, -1);
}

void CheckTransitions::preGlobalEndLumi(GlobalContext const& gc) {
  auto id = gc.luminosityBlockID();
  m_seenTransitions.emplace_back(Phase::kEndLumi, edm::EventID{id.run(), id.luminosityBlock(), 0}, 1000);
}

void CheckTransitions::preStreamBeginLumi(StreamContext const& sc) {
  m_seenTransitions.emplace_back(Phase::kBeginLumi, sc.eventID(), sc.streamID());
}

void CheckTransitions::preStreamEndLumi(StreamContext const& sc) {
  m_seenTransitions.emplace_back(Phase::kEndLumi, sc.eventID(), sc.streamID());
}

void CheckTransitions::preEvent(StreamContext const& sc) {
  m_seenTransitions.emplace_back(Phase::kEvent, sc.eventID(), sc.streamID());
}

using edm::service::CheckTransitions;
DEFINE_FWK_SERVICE(CheckTransitions);
