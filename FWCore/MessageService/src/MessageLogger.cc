// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  W. Brown, M. Fischler
//         Created:  Fri Nov 11 16:42:39 CST 2005
//

// system include files
// user include files

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageService/interface/MessageLogger.h"
#include "FWCore/MessageService/src/MessageServicePSetValidation.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"

#include <sstream>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <charconv>
#include <cassert>

using namespace edm;
using namespace edm::service;

namespace {
  constexpr std::array<char const*, 12> s_globalTransitionNames = {{"@beginJob",
                                                                    "@beginProcessBlock",
                                                                    "@accessInputProcessBlock",
                                                                    "@beginRun",
                                                                    "@beginLumi",
                                                                    "@endLumi",
                                                                    "@endRun",
                                                                    "@kEndProcessBlock",
                                                                    "@endJob",
                                                                    "@writeProcessBlock",
                                                                    "@writeRun",
                                                                    "@writeLumi"}};

  constexpr std::array<char const*, 7> s_streamTransitionNames = {{"@beginStream",
                                                                   "@streamBeginRun",
                                                                   "@streamBeginLumi",
                                                                   "",  //event
                                                                   "@streamEndLumi",
                                                                   "@streamEndRun",
                                                                   "@endStream"}};

  char* fill_buffer(char* p, char*) { return p; }

  template <typename T, typename... U>
  char* fill_buffer(char* first, char* last, T value, U... u) {
    if constexpr (std::is_arithmetic<T>::value) {
      auto v = std::to_chars(first, last, value);
      assert(v.ec == std::errc{});
      return fill_buffer(v.ptr, last, std::forward<U>(u)...);
    } else {
      auto l = strlen(value);
      assert(first + l < last);
      std::copy(value, value + l, first);
      return fill_buffer(first + l, last, std::forward<U>(u)...);
    }
  }

  template <typename... T>
  std::string_view fill_buffer(std::array<char, 64>& buffer, T... t) {
    auto e = fill_buffer(buffer.begin(), buffer.end(), std::forward<T>(t)...);
    assert(e < buffer.end());
    *e = 0;
    return std::string_view(buffer.begin(), e - buffer.begin() + 1);
  }

}  // namespace

namespace edm {
  //Forward declare here
  // Only the MessageLogger::postEVent function is allowed to call this function.
  // So although the function is defined in MessageSender.cc this is the
  //  only place where we want it declared.
  void clearLoggedErrorsSummary(unsigned int);
  void setMaxLoggedErrorsSummaryIndicies(unsigned int iMax);

  namespace service {

    bool edm::service::MessageLogger::anyDebugEnabled_ = false;
    bool edm::service::MessageLogger::everyDebugEnabled_ = false;
    bool edm::service::MessageLogger::fjrSummaryRequested_ = false;

    //
    // constructors and destructor
    //
    edm::service::MessageLogger::MessageLogger(ParameterSet const& iPS, ActivityRegistry& iRegistry)
        : debugEnabled_(false),
          messageServicePSetHasBeenValidated_(false),
          messageServicePSetValidatationResults_(),
          nonModule_debugEnabled(false),
          nonModule_infoEnabled(true),
          nonModule_warningEnabled(true),
          nonModule_errorEnabled(true)  // change log 20
    {
      // prepare cfg validation string for later use
      MessageServicePSetValidation validator;
      messageServicePSetValidatationResults_ = validator(iPS);  // change log 12

      typedef std::vector<std::string> vString;
      vString empty_vString;
      vString debugModules;
      vString suppressDebug;
      vString suppressInfo;
      vString suppressFwkInfo;
      vString suppressWarning;
      vString suppressError;  // change log 20

      try {  // change log 13
             // decide whether a summary should be placed in job report
        fjrSummaryRequested_ = iPS.getUntrackedParameter<bool>("messageSummaryToJobReport", false);

        // grab list of debug-enabled modules
        debugModules = iPS.getUntrackedParameter<vString>("debugModules", empty_vString);

        // grab lists of suppressLEVEL modules
        suppressDebug = iPS.getUntrackedParameter<vString>("suppressDebug", empty_vString);

        suppressInfo = iPS.getUntrackedParameter<vString>("suppressInfo", empty_vString);

        suppressFwkInfo = iPS.getUntrackedParameter<vString>("suppressFwkInfo", empty_vString);

        suppressWarning = iPS.getUntrackedParameter<vString>("suppressWarning", empty_vString);

        suppressError =  // change log 20
            iPS.getUntrackedParameter<vString>("suppressError", empty_vString);
      } catch (cms::Exception& e) {  // change log 13
      }

      // Use these lists to prepare a map to use in tracking suppression

      // Do suppressDebug first and suppressError last to get proper order
      for (vString::const_iterator it = suppressDebug.begin(); it != suppressDebug.end(); ++it) {
        suppression_levels_[*it] = ELseverityLevel::ELsev_success;
      }

      for (vString::const_iterator it = suppressInfo.begin(); it != suppressInfo.end(); ++it) {
        suppression_levels_[*it] = ELseverityLevel::ELsev_info;
      }

      for (vString::const_iterator it = suppressFwkInfo.begin(); it != suppressFwkInfo.end(); ++it) {
        suppression_levels_[*it] = ELseverityLevel::ELsev_fwkInfo;
      }

      for (vString::const_iterator it = suppressWarning.begin(); it != suppressWarning.end(); ++it) {
        suppression_levels_[*it] = ELseverityLevel::ELsev_warning;
      }

      for (vString::const_iterator it = suppressError.begin();  // change log 20
           it != suppressError.end();
           ++it) {
        suppression_levels_[*it] = ELseverityLevel::ELsev_error;
      }

      // set up for tracking whether current module is debug-enabled
      // (and info-enabled and warning-enabled)
      if (debugModules.empty()) {
        anyDebugEnabled_ = false;                       // change log 11
        MessageDrop::instance()->debugEnabled = false;  // change log 1
      } else {
        anyDebugEnabled_ = true;  // change log 11
        MessageDrop::instance()->debugEnabled = false;
        // this will be over-ridden when specific modules are entered
      }

      // if ( debugModules.empty()) anyDebugEnabled_ = true; // wrong; change log 11
      for (vString::const_iterator it = debugModules.begin(); it != debugModules.end(); ++it) {
        if (*it == "*") {
          everyDebugEnabled_ = true;
        } else {
          debugEnabledModules_.insert(*it);
        }
      }

      // change log 7
      std::string jm = edm::MessageDrop::jobMode;
      std::string* jm_p = new std::string(jm);
      MessageLoggerQ::MLqMOD(jm_p);  // change log 9

      MessageLoggerQ::MLqCFG(new ParameterSet(iPS));  // change log 9

      iRegistry.watchPreallocate([this](edm::service::SystemBounds const& iBounds) {
        //reserve the proper amount of space to record the transition info
        this->transitionInfoCache_.resize(iBounds.maxNumberOfStreams() +
                                          iBounds.maxNumberOfConcurrentLuminosityBlocks() +
                                          iBounds.maxNumberOfConcurrentRuns());
        lumiInfoBegin_ = iBounds.maxNumberOfStreams();
        runInfoBegin_ = lumiInfoBegin_ + iBounds.maxNumberOfConcurrentLuminosityBlocks();

        setMaxLoggedErrorsSummaryIndicies(iBounds.maxNumberOfStreams());
      });

      iRegistry.watchPostBeginJob(this, &MessageLogger::postBeginJob);
      iRegistry.watchPreEndJob(this, &MessageLogger::preEndJob);
      iRegistry.watchPostEndJob(this, &MessageLogger::postEndJob);
      iRegistry.watchJobFailure(this, &MessageLogger::jobFailure);  // change log 14

      iRegistry.watchPreModuleConstruction(this, &MessageLogger::preModuleConstruction);
      iRegistry.watchPostModuleConstruction(this, &MessageLogger::postModuleConstruction);
      // change log 3

      iRegistry.watchPreSourceConstruction(this, &MessageLogger::preSourceConstruction);
      iRegistry.watchPostSourceConstruction(this, &MessageLogger::postSourceConstruction);
      // change log 3

      iRegistry.watchPreModuleEvent(this, &MessageLogger::preModuleEvent);
      iRegistry.watchPostModuleEvent(this, &MessageLogger::postModuleEvent);

      iRegistry.watchPreModuleEventAcquire(this, &MessageLogger::preModuleEventAcquire);
      iRegistry.watchPostModuleEventAcquire(this, &MessageLogger::postModuleEventAcquire);

      iRegistry.watchPreSourceEvent(this, &MessageLogger::preSourceEvent);
      iRegistry.watchPostSourceEvent(this, &MessageLogger::postSourceEvent);
      // change log 14:
      iRegistry.watchPreSourceRun([this](RunIndex) { preSourceRunLumi(); });
      iRegistry.watchPostSourceRun([this](RunIndex) { postSourceRunLumi(); });
      iRegistry.watchPreSourceLumi([this](LuminosityBlockIndex) { preSourceRunLumi(); });
      iRegistry.watchPostSourceLumi([this](LuminosityBlockIndex) { postSourceRunLumi(); });
      iRegistry.watchPreOpenFile(this, &MessageLogger::preFile);
      iRegistry.watchPostOpenFile(this, &MessageLogger::postFile);
      iRegistry.watchPreCloseFile(this, &MessageLogger::preFileClose);
      iRegistry.watchPostCloseFile(this, &MessageLogger::postFile);

      // change log 13:
      // change log 15
      iRegistry.watchPreModuleBeginJob(this, &MessageLogger::preModuleBeginJob);
      iRegistry.watchPostModuleBeginJob(this, &MessageLogger::postModuleBeginJob);
      iRegistry.watchPreModuleEndJob(this, &MessageLogger::preModuleEndJob);
      iRegistry.watchPostModuleEndJob(this, &MessageLogger::postModuleEndJob);

      iRegistry.watchPreModuleBeginStream(this, &MessageLogger::preModuleBeginStream);
      iRegistry.watchPostModuleBeginStream(this, &MessageLogger::postModuleBeginStream);
      iRegistry.watchPreModuleEndStream(this, &MessageLogger::preModuleEndStream);
      iRegistry.watchPostModuleEndStream(this, &MessageLogger::postModuleEndStream);

      iRegistry.watchPreModuleStreamBeginRun(this, &MessageLogger::preModuleStreamBeginRun);
      iRegistry.watchPostModuleStreamBeginRun(this, &MessageLogger::postModuleStreamBeginRun);
      iRegistry.watchPreModuleStreamEndRun(this, &MessageLogger::preModuleStreamEndRun);
      iRegistry.watchPostModuleStreamEndRun(this, &MessageLogger::postModuleStreamEndRun);
      iRegistry.watchPreModuleStreamBeginLumi(this, &MessageLogger::preModuleStreamBeginLumi);
      iRegistry.watchPostModuleStreamBeginLumi(this, &MessageLogger::postModuleStreamBeginLumi);
      iRegistry.watchPreModuleStreamEndLumi(this, &MessageLogger::preModuleStreamEndLumi);
      iRegistry.watchPostModuleStreamEndLumi(this, &MessageLogger::postModuleStreamEndLumi);

      iRegistry.watchPreModuleBeginProcessBlock(this, &MessageLogger::preModuleBeginProcessBlock);
      iRegistry.watchPostModuleBeginProcessBlock(this, &MessageLogger::postModuleBeginProcessBlock);
      iRegistry.watchPreModuleAccessInputProcessBlock(this, &MessageLogger::preModuleAccessInputProcessBlock);
      iRegistry.watchPostModuleAccessInputProcessBlock(this, &MessageLogger::postModuleAccessInputProcessBlock);
      iRegistry.watchPreModuleEndProcessBlock(this, &MessageLogger::preModuleEndProcessBlock);
      iRegistry.watchPostModuleEndProcessBlock(this, &MessageLogger::postModuleEndProcessBlock);

      iRegistry.watchPreModuleGlobalBeginRun(this, &MessageLogger::preModuleGlobalBeginRun);
      iRegistry.watchPostModuleGlobalBeginRun(this, &MessageLogger::postModuleGlobalBeginRun);
      iRegistry.watchPreModuleGlobalEndRun(this, &MessageLogger::preModuleGlobalEndRun);
      iRegistry.watchPostModuleGlobalEndRun(this, &MessageLogger::postModuleGlobalEndRun);
      iRegistry.watchPreModuleGlobalBeginLumi(this, &MessageLogger::preModuleGlobalBeginLumi);
      iRegistry.watchPostModuleGlobalBeginLumi(this, &MessageLogger::postModuleGlobalBeginLumi);
      iRegistry.watchPreModuleGlobalEndLumi(this, &MessageLogger::preModuleGlobalEndLumi);
      iRegistry.watchPostModuleGlobalEndLumi(this, &MessageLogger::postModuleGlobalEndLumi);

      iRegistry.watchPreEvent(this, &MessageLogger::preEvent);
      iRegistry.watchPostEvent(this, &MessageLogger::postEvent);

      iRegistry.watchPreStreamBeginRun(this, &MessageLogger::preStreamBeginRun);
      iRegistry.watchPostStreamBeginRun(this, &MessageLogger::postStreamBeginRun);
      iRegistry.watchPreStreamEndRun(this, &MessageLogger::preStreamEndRun);
      iRegistry.watchPostStreamEndRun(this, &MessageLogger::postStreamEndRun);
      iRegistry.watchPreStreamBeginLumi(this, &MessageLogger::preStreamBeginLumi);
      iRegistry.watchPostStreamBeginLumi(this, &MessageLogger::postStreamBeginLumi);
      iRegistry.watchPreStreamEndLumi(this, &MessageLogger::preStreamEndLumi);
      iRegistry.watchPostStreamEndLumi(this, &MessageLogger::postStreamEndLumi);

      iRegistry.watchPreBeginProcessBlock(this, &MessageLogger::preBeginProcessBlock);
      iRegistry.watchPostBeginProcessBlock(this, &MessageLogger::postBeginProcessBlock);
      iRegistry.watchPreAccessInputProcessBlock(this, &MessageLogger::preAccessInputProcessBlock);
      iRegistry.watchPostAccessInputProcessBlock(this, &MessageLogger::postAccessInputProcessBlock);
      iRegistry.watchPreEndProcessBlock(this, &MessageLogger::preEndProcessBlock);
      iRegistry.watchPostEndProcessBlock(this, &MessageLogger::postEndProcessBlock);

      iRegistry.watchPreGlobalBeginRun(this, &MessageLogger::preGlobalBeginRun);
      iRegistry.watchPostGlobalBeginRun(this, &MessageLogger::postGlobalBeginRun);
      iRegistry.watchPreGlobalEndRun(this, &MessageLogger::preGlobalEndRun);
      iRegistry.watchPostGlobalEndRun(this, &MessageLogger::postGlobalEndRun);
      iRegistry.watchPreGlobalBeginLumi(this, &MessageLogger::preGlobalBeginLumi);
      iRegistry.watchPostGlobalBeginLumi(this, &MessageLogger::postGlobalBeginLumi);
      iRegistry.watchPreGlobalEndLumi(this, &MessageLogger::preGlobalEndLumi);
      iRegistry.watchPostGlobalEndLumi(this, &MessageLogger::postGlobalEndLumi);

      iRegistry.watchPrePathEvent(this, &MessageLogger::prePathEvent);
      iRegistry.watchPostPathEvent(this, &MessageLogger::postPathEvent);

      MessageDrop* messageDrop = MessageDrop::instance();
      nonModule_debugEnabled = messageDrop->debugEnabled;
      nonModule_infoEnabled = messageDrop->infoEnabled;
      nonModule_warningEnabled = messageDrop->warningEnabled;
      nonModule_errorEnabled = messageDrop->errorEnabled;
    }  // ctor

    //
    // Shared helper routines for establishing module name and enabling behavior
    //

    void MessageLogger::establishModule(ModuleDescription const& desc,
                                        const char* whichPhase)  // ChangeLog 13, 17
    {
      MessageDrop* messageDrop = MessageDrop::instance();

      // std::cerr << "establishModule( " << desc.moduleName() << ")\n";
      // Change Log 17
      messageDrop->setModuleWithPhase(desc.moduleName(), desc.moduleLabel(), desc.id(), whichPhase);
      // Removed caching per change 17 - caching is now done in MessageDrop.cc
      // in theContext() method, and only happens if a message is actually issued.

      if (!anyDebugEnabled_) {
        messageDrop->debugEnabled = false;
      } else if (everyDebugEnabled_) {
        messageDrop->debugEnabled = true;
      } else {
        messageDrop->debugEnabled = debugEnabledModules_.count(desc.moduleLabel());
      }

      auto it = suppression_levels_.find(desc.moduleLabel());
      if (it != suppression_levels_.end()) {
        messageDrop->debugEnabled = messageDrop->debugEnabled && (it->second < ELseverityLevel::ELsev_success);
        messageDrop->infoEnabled = (it->second < ELseverityLevel::ELsev_info);
        messageDrop->fwkInfoEnabled = (it->second < ELseverityLevel::ELsev_fwkInfo);
        messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning);
        messageDrop->errorEnabled = (it->second < ELseverityLevel::ELsev_error);
      } else {
        messageDrop->infoEnabled = true;
        messageDrop->fwkInfoEnabled = true;
        messageDrop->warningEnabled = true;
        messageDrop->errorEnabled = true;
      }
    }  // establishModule

    void MessageLogger::establishModule(unsigned int transitionIndex,
                                        ModuleCallingContext const& mod,
                                        const char* whichPhase)  // ChangeLog 13, 17
    {
      MessageDrop* messageDrop = MessageDrop::instance();

      // std::cerr << "establishModule( " << desc.moduleName() << ")\n";
      // Change Log 17
      auto const desc = mod.moduleDescription();
      messageDrop->runEvent = transitionInfoCache_[transitionIndex].begin();
      messageDrop->setModuleWithPhase(desc->moduleName(), desc->moduleLabel(), desc->id(), whichPhase);
      messageDrop->streamID = transitionIndex;
      if (transitionIndex >= lumiInfoBegin_) {
        messageDrop->streamID = std::numeric_limits<unsigned int>::max();
      }
      // Removed caching per change 17 - caching is now done in MessageDrop.cc
      // in theContext() method, and only happens if a message is actually issued.

      if (!anyDebugEnabled_) {
        messageDrop->debugEnabled = false;
      } else if (everyDebugEnabled_) {
        messageDrop->debugEnabled = true;
      } else {
        messageDrop->debugEnabled = debugEnabledModules_.count(desc->moduleLabel());
      }

      auto it = suppression_levels_.find(desc->moduleLabel());
      if (it != suppression_levels_.end()) {
        messageDrop->debugEnabled = messageDrop->debugEnabled && (it->second < ELseverityLevel::ELsev_success);
        messageDrop->infoEnabled = (it->second < ELseverityLevel::ELsev_info);
        messageDrop->fwkInfoEnabled = (it->second < ELseverityLevel::ELsev_fwkInfo);
        messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning);
        messageDrop->errorEnabled = (it->second < ELseverityLevel::ELsev_error);
      } else {
        messageDrop->infoEnabled = true;
        messageDrop->fwkInfoEnabled = true;
        messageDrop->warningEnabled = true;
        messageDrop->errorEnabled = true;
      }
    }  // establishModule

    void MessageLogger::unEstablishModule(ModuleDescription const& /*unused*/, const char* state) {
      // std::cerr << "unestablishModule( " << desc.moduleName() << ") "
      //           << "state = " << *state << "\n";

      MessageDrop* messageDrop = MessageDrop::instance();
      messageDrop->setSinglet(state);  // Change Log 17
      messageDrop->debugEnabled = nonModule_debugEnabled;
      messageDrop->infoEnabled = nonModule_infoEnabled;
      messageDrop->warningEnabled = nonModule_warningEnabled;
      messageDrop->errorEnabled = nonModule_errorEnabled;  // change log 20
    }

    void MessageLogger::unEstablishModule(ModuleCallingContext const& mod, const char* state) {
      //Need to reset to what was previously being used on this thread
      auto previous = mod.previousModuleOnThread();
      if (previous) {
        //need to know if we are in a global or stream context
        auto top = previous->getTopModuleCallingContext();
        assert(nullptr != top);
        if (ParentContext::Type::kGlobal == top->type()) {
          auto globalContext = top->globalContext();
          assert(nullptr != globalContext);
          auto tran = globalContext->transition();
          if (tran == GlobalContext::Transition::kBeginLuminosityBlock or
              tran == GlobalContext::Transition::kEndLuminosityBlock) {
            establishModule(lumiInfoBegin_ + globalContext->luminosityBlockIndex(),
                            *previous,
                            s_globalTransitionNames[static_cast<int>(tran)]);
          } else {
            establishModule(
                runInfoBegin_ + globalContext->runIndex(), *previous, s_globalTransitionNames[static_cast<int>(tran)]);
          }
        } else {
          auto stream = previous->getStreamContext();
          establishModule(
              stream->streamID().value(), *previous, s_streamTransitionNames[static_cast<int>(stream->transition())]);
        }
      } else {
        MessageDrop* messageDrop = MessageDrop::instance();
        messageDrop->streamID = std::numeric_limits<unsigned int>::max();
        messageDrop->setSinglet(state);  // Change Log 17
        messageDrop->debugEnabled = nonModule_debugEnabled;
        messageDrop->infoEnabled = nonModule_infoEnabled;
        messageDrop->warningEnabled = nonModule_warningEnabled;
        messageDrop->errorEnabled = nonModule_errorEnabled;  // change log 20
      }

      // std::cerr << "unestablishModule( " << desc.moduleName() << ") "
      //           << "state = " << *state << "\n";
    }

    void MessageLogger::establish(const char* state) {
      MessageDrop* messageDrop = MessageDrop::instance();
      messageDrop->setSinglet(state);  // Change Log 17
      if (!anyDebugEnabled_) {
        messageDrop->debugEnabled = false;
      } else if (everyDebugEnabled_) {
        messageDrop->debugEnabled = true;
      } else {
        messageDrop->debugEnabled = debugEnabledModules_.count(state);  // change log 8
      }
      std::map<const std::string, ELseverityLevel>::const_iterator it =
          suppression_levels_.find(state);  // change log 8
      if (it != suppression_levels_.end()) {
        messageDrop->debugEnabled = messageDrop->debugEnabled && (it->second < ELseverityLevel::ELsev_success);
        messageDrop->infoEnabled = (it->second < ELseverityLevel::ELsev_info);
        messageDrop->fwkInfoEnabled = (it->second < ELseverityLevel::ELsev_fwkInfo);
        messageDrop->warningEnabled = (it->second < ELseverityLevel::ELsev_warning);
        messageDrop->errorEnabled = (it->second < ELseverityLevel::ELsev_error);
      } else {
        messageDrop->infoEnabled = true;
        messageDrop->fwkInfoEnabled = true;
        messageDrop->warningEnabled = true;
        messageDrop->errorEnabled = true;
      }
    }

    void MessageLogger::unEstablish(const char* state) {
      MessageDrop::instance()->setSinglet(state);  // Change Log 17
    }

    //
    // callbacks that need to establish the module, and their counterparts
    //

    void MessageLogger::preModuleConstruction(const ModuleDescription& desc) {
      if (!messageServicePSetHasBeenValidated_) {  // change log 12
        if (!messageServicePSetValidatationResults_.empty()) {
          throw(edm::Exception(edm::errors::Configuration, messageServicePSetValidatationResults_));
        }
        messageServicePSetHasBeenValidated_ = true;
      }
      establishModule(desc, "@ctor");  // ChangeLog 16
    }
    void MessageLogger::postModuleConstruction(
        const ModuleDescription&
            iDescription) {  //it is now guaranteed that this will be called even if the module throws
      unEstablishModule(iDescription, "AfterModConstruction");
    }

    void MessageLogger::preModuleBeginJob(const ModuleDescription& desc) {
      establishModule(desc, "@beginJob");  // ChangeLog 13
    }
    void MessageLogger::postModuleBeginJob(const ModuleDescription& iDescription) {
      unEstablishModule(iDescription, "AfterModBeginJob");
    }

    void MessageLogger::preSourceConstruction(const ModuleDescription& desc) {
      if (!messageServicePSetHasBeenValidated_) {  // change log 12
        if (!messageServicePSetValidatationResults_.empty()) {
          throw(edm::Exception(edm::errors::Configuration, messageServicePSetValidatationResults_));
        }
        messageServicePSetHasBeenValidated_ = true;
      }
      establishModule(desc, "@sourceConstruction");  // ChangeLog 16
    }
    void MessageLogger::postSourceConstruction(const ModuleDescription& iDescription) {
      unEstablishModule(iDescription, "AfterSourceConstruction");
    }

    void MessageLogger::preModuleBeginStream(StreamContext const& stream, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      establishModule(desc, "@beginStream");  // ChangeLog 13
    }
    void MessageLogger::postModuleBeginStream(StreamContext const& stream, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      unEstablishModule(desc, "AfterModBeginStream");
    }

    void MessageLogger::preModuleStreamBeginRun(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(stream.streamID().value(),
                      mod,
                      s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kBeginRun)]);
    }
    void MessageLogger::postModuleStreamBeginRun(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModStreamBeginRun");
    }

    void MessageLogger::preModuleStreamBeginLumi(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(stream.streamID().value(),
                      mod,
                      s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kBeginLuminosityBlock)]);
    }
    void MessageLogger::postModuleStreamBeginLumi(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModStreamBeginLumi");
    }

    void MessageLogger::preModuleEvent(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(
          stream.streamID().value(), mod, s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEvent)]);
    }

    void MessageLogger::postModuleEvent(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "PostModuleEvent");
    }

    void MessageLogger::preModuleEventAcquire(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(
          stream.streamID().value(), mod, s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEvent)]);
    }

    void MessageLogger::postModuleEventAcquire(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "PostModuleEventAcquire");
    }

    void MessageLogger::preModuleStreamEndLumi(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(stream.streamID().value(),
                      mod,
                      s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEndLuminosityBlock)]);
    }
    void MessageLogger::postModuleStreamEndLumi(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModStreamEndLumi");
    }

    void MessageLogger::preModuleStreamEndRun(StreamContext const& stream, ModuleCallingContext const& mod) {
      establishModule(stream.streamID().value(),
                      mod,
                      s_streamTransitionNames[static_cast<int>(StreamContext::Transition::kEndRun)]);  // ChangeLog 13
    }
    void MessageLogger::postModuleStreamEndRun(StreamContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModStreamEndRun");
    }

    //Global

    void MessageLogger::preModuleBeginProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      establishModule(desc, "@beginProcessBlock");
    }

    void MessageLogger::postModuleBeginProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      unEstablishModule(desc, "After module BeginProcessBlock");
    }

    void MessageLogger::preModuleAccessInputProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      establishModule(desc, "@accessInputProcessBlock");
    }

    void MessageLogger::postModuleAccessInputProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      unEstablishModule(desc, "After module AccessInputProcessBlock");
    }

    void MessageLogger::preModuleEndProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      establishModule(desc, "@endProcessBlock");
    }

    void MessageLogger::postModuleEndProcessBlock(GlobalContext const& gc, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      unEstablishModule(desc, "After module EndProcessBlock");
    }

    void MessageLogger::preModuleGlobalBeginRun(GlobalContext const& context, ModuleCallingContext const& mod) {
      establishModule(runInfoBegin_ + context.runIndex().value(),
                      mod,
                      s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kBeginRun)]);
    }
    void MessageLogger::postModuleGlobalBeginRun(GlobalContext const& context, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModGlobalBeginRun");
    }

    void MessageLogger::preModuleGlobalBeginLumi(GlobalContext const& context, ModuleCallingContext const& mod) {
      establishModule(lumiInfoBegin_ + context.luminosityBlockIndex().value(),
                      mod,
                      s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kBeginLuminosityBlock)]);
    }
    void MessageLogger::postModuleGlobalBeginLumi(GlobalContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModGlobalBeginLumi");
    }

    void MessageLogger::preModuleGlobalEndLumi(GlobalContext const& context, ModuleCallingContext const& mod) {
      establishModule(lumiInfoBegin_ + context.luminosityBlockIndex().value(),
                      mod,
                      s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kEndLuminosityBlock)]);
    }
    void MessageLogger::postModuleGlobalEndLumi(GlobalContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModGlobalEndLumi");
    }

    void MessageLogger::preModuleGlobalEndRun(GlobalContext const& context, ModuleCallingContext const& mod) {
      establishModule(runInfoBegin_ + context.runIndex().value(),
                      mod,
                      s_globalTransitionNames[static_cast<int>(GlobalContext::Transition::kEndRun)]);  // ChangeLog 13
    }
    void MessageLogger::postModuleGlobalEndRun(GlobalContext const& stream, ModuleCallingContext const& mod) {
      unEstablishModule(mod, "AfterModGlobalEndRun");
    }

    void MessageLogger::preModuleEndStream(StreamContext const&, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      establishModule(desc, "@endStream");  // ChangeLog 13
    }

    void MessageLogger::postModuleEndStream(StreamContext const&, ModuleCallingContext const& mcc) {
      ModuleDescription const& desc = *mcc.moduleDescription();
      unEstablishModule(desc, "AfterModEndStream");
    }

    void MessageLogger::preModuleEndJob(const ModuleDescription& desc) {
      establishModule(desc, "@endJob");  // ChangeLog 13
    }
    void MessageLogger::postModuleEndJob(const ModuleDescription& iDescription) {
      unEstablishModule(iDescription, "AfterModEndJob");
    }

    //
    // callbacks that don't know about the module
    //

    void MessageLogger::postBeginJob() {
      MessageDrop::instance()->runEvent = "BeforeEvents";
      MessageDrop::instance()->setSinglet("AfterBeginJob");  // Change Log 17
    }

    void MessageLogger::preSourceEvent(StreamID) {
      establish("source");
      MessageDrop::instance()->runEvent = "PreSource";
    }
    void MessageLogger::postSourceEvent(StreamID) {
      unEstablish("AfterSource");
      MessageDrop::instance()->runEvent = "AfterSource";
    }
    void MessageLogger::preSourceRunLumi() { establish("source"); }
    void MessageLogger::postSourceRunLumi() { unEstablish("AfterSource"); }

    void MessageLogger::preFile(std::string const&, bool) { establish("file_open"); }
    void MessageLogger::preFileClose(std::string const&, bool) { establish("file_close"); }
    void MessageLogger::postFile(std::string const&, bool) { unEstablish("AfterFile"); }

    void MessageLogger::preEvent(StreamContext const& iContext) {
      assert(iContext.streamID().value() < transitionInfoCache_.size());
      auto& buffer = transitionInfoCache_[iContext.streamID().value()];
      auto const& id = iContext.eventID();
      auto v = fill_buffer(buffer, "Run: ", id.run(), " Event: ", id.event());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreEventProcessing");  // changelog 17
          // Note - module name had not been set here  Similarly in other places where
          // RunEvent carries the new information; we add setSinglet for module name.
    }

    void MessageLogger::postEvent(StreamContext const& iContext) {
      edm::MessageDrop::instance()->runEvent = "PostProcessEvent";
      edm::clearLoggedErrorsSummary(iContext.streamID().value());
    }

    void MessageLogger::preStreamBeginRun(StreamContext const& iContext)  // change log 14
    {
      auto& buffer = transitionInfoCache_[iContext.streamID().value()];
      auto v = fill_buffer(buffer, "Run: ", iContext.eventID().run(), " Stream: ", iContext.streamID().value());

      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreStreamBeginRun");  // changelog 17
    }
    void MessageLogger::postStreamBeginRun(StreamContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostStreamBeginRun";
      edm::MessageDrop::instance()->setSinglet("PostStreamBeginRun");  // changelog 17
                                                                       // Note - module name had not been set here
    }

    void MessageLogger::preStreamEndRun(StreamContext const& iContext) {
      auto& buffer = transitionInfoCache_[iContext.streamID().value()];
      auto v = fill_buffer(buffer, "End Run: ", iContext.eventID().run(), " Stream: ", iContext.streamID().value());

      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreStreamEndRun");  // changelog 17
    }

    void MessageLogger::postStreamEndRun(StreamContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostStreamEndRun";
      edm::MessageDrop::instance()->setSinglet("PostStreaEndRun");  // changelog 17
    }

    void MessageLogger::preStreamBeginLumi(StreamContext const& iContext) {
      auto& buffer = transitionInfoCache_[iContext.streamID().value()];
      auto const& id = iContext.eventID();
      auto v = fill_buffer(
          buffer, "Run: ", id.run(), " Lumi: ", id.luminosityBlock(), " Stream: ", iContext.streamID().value());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreStreamBeginLumi");  // changelog 17
    }

    void MessageLogger::postStreamBeginLumi(StreamContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostStreamBeginLumi";
      edm::MessageDrop::instance()->setSinglet("PostStreamBeginLumi");  // changelog 17
    }

    void MessageLogger::preStreamEndLumi(StreamContext const& iContext) {
      auto& buffer = transitionInfoCache_[iContext.streamID().value()];
      auto const& id = iContext.eventID();
      auto v = fill_buffer(
          buffer, "Run: ", id.run(), " Lumi: ", id.luminosityBlock(), " Stream: ", iContext.streamID().value());

      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreStreamEndLumi");  // changelog 17
    }
    void MessageLogger::postStreamEndLumi(StreamContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostStreamEndLumi";
      edm::MessageDrop::instance()->setSinglet("PostStreamEndLumi");  // changelog 17
    }

    void MessageLogger::preBeginProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->runEvent = "pre-events";
      edm::MessageDrop::instance()->setSinglet("BeginProcessBlock");
    }

    void MessageLogger::postBeginProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->setSinglet("After BeginProcessBlock");
    }

    void MessageLogger::preAccessInputProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->runEvent = "pre-events";
      edm::MessageDrop::instance()->setSinglet("AccessInputProcessBlock");
    }

    void MessageLogger::postAccessInputProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->setSinglet("After AccessInputProcessBlock");
    }

    void MessageLogger::preEndProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->runEvent = "post-events";
      edm::MessageDrop::instance()->setSinglet("EndProcessBlock");
    }

    void MessageLogger::postEndProcessBlock(GlobalContext const& gc) {
      edm::MessageDrop::instance()->setSinglet("After EndProcessBlock");
    }

    void MessageLogger::preGlobalBeginRun(GlobalContext const& iContext)  // change log 14
    {
      auto& buffer = transitionInfoCache_[runInfoBegin_ + iContext.runIndex()];
      auto v = fill_buffer(buffer, "Run: ", iContext.luminosityBlockID().run());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreGlobalBeginRun");  // changelog 17
    }
    void MessageLogger::postGlobalBeginRun(GlobalContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostGlobalBeginRun";
      edm::MessageDrop::instance()->setSinglet("PostGlobalBeginRun");  // changelog 17
                                                                       // Note - module name had not been set here
    }

    void MessageLogger::prePathEvent(StreamContext const& stream, PathContext const& iPath)  // change log 14
    {
      auto messageDrop = edm::MessageDrop::instance();
      messageDrop->runEvent = transitionInfoCache_[stream.streamID().value()].begin();
      messageDrop->setPath("PreProcPath ", iPath.pathName());
      // change log 17
    }

    void MessageLogger::postPathEvent(StreamContext const&, PathContext const&, HLTPathStatus const&) {
      edm::MessageDrop::instance()->setSinglet("PostProcessPath");  // changelog 17
    }

    void MessageLogger::preGlobalEndRun(GlobalContext const& iContext) {
      auto& buffer = transitionInfoCache_[runInfoBegin_ + iContext.runIndex()];
      auto v = fill_buffer(buffer, "End Run: ", iContext.luminosityBlockID().run());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreGlobalEndRun");  // changelog 17
    }

    void MessageLogger::postGlobalEndRun(GlobalContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostGlobalEndRun";
      edm::MessageDrop::instance()->setSinglet("PostGlobalEndRun");  // changelog 17
    }

    void MessageLogger::preGlobalBeginLumi(GlobalContext const& iContext) {
      auto& buffer = transitionInfoCache_[lumiInfoBegin_ + iContext.luminosityBlockIndex()];
      auto const& id = iContext.luminosityBlockID();
      auto v = fill_buffer(buffer, "Run: ", id.run(), " Lumi: ", id.luminosityBlock());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreGlobalBeginLumi");  // changelog 17
    }

    void MessageLogger::postGlobalBeginLumi(GlobalContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostGlobalBeginLumi";
      edm::MessageDrop::instance()->setSinglet("PostGlobalBeginLumi");  // changelog 17
    }

    void MessageLogger::preGlobalEndLumi(GlobalContext const& iContext) {
      auto& buffer = transitionInfoCache_[lumiInfoBegin_ + iContext.luminosityBlockIndex()];
      auto const& id = iContext.luminosityBlockID();
      auto v = fill_buffer(buffer, "Run: ", id.run(), " Lumi: ", id.luminosityBlock());
      edm::MessageDrop::instance()->runEvent = v;
      edm::MessageDrop::instance()->setSinglet("PreGlobalEndLumi");  // changelog 17
    }
    void MessageLogger::postGlobalEndLumi(GlobalContext const&) {
      edm::MessageDrop::instance()->runEvent = "PostGlobalEndLumi";
      edm::MessageDrop::instance()->setSinglet("PostGlobalEndLumi");  // changelog 17
    }

    void MessageLogger::preEndJob() {
      edm::MessageDrop::instance()->runEvent = "EndJob";
      edm::MessageDrop::instance()->setSinglet("EndJob");  // changelog
    }

    void MessageLogger::postEndJob() {
      SummarizeInJobReport();    // Put summary info into Job Rep  // change log 10
      MessageLoggerQ::MLqSUM();  // trigger summary info.		// change log 9
    }

    void MessageLogger::jobFailure() {
      MessageDrop* messageDrop = MessageDrop::instance();
      messageDrop->setSinglet("jobFailure");
      SummarizeInJobReport();    // Put summary info into Job Rep  // change log 10
      MessageLoggerQ::MLqSUM();  // trigger summary info.		// change log 9
    }

    //
    // Other methods
    //

    void MessageLogger::SummarizeInJobReport() {
      if (fjrSummaryRequested_) {
        std::map<std::string, double>* smp = new std::map<std::string, double>();
        MessageLoggerQ::MLqJRS(smp);
        Service<JobReport> reportSvc;
        reportSvc->reportMessageInfo(*smp);
        delete smp;
      }
    }

  }  // end of namespace service
}  // end of namespace edm
