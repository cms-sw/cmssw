// -*- C++ -*-
//
// Package:     MessageLogger
// Class  :     MessageDrop
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  M. Fischler and Jim Kowalkowsi
//         Created:  Tues Feb 14 16:38:19 CST 2006
//

// system include files
#include <cstring>
#include <limits>

// user include files
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

using namespace edm;

// The following are false at initialization (in case configure is not done)
// and are set true at the start of configure_ordinary_destinations,
// but are set false once a destination is thresholded to react to the
// corresponding severity:
bool MessageDrop::debugAlwaysSuppressed = false;
bool MessageDrop::infoAlwaysSuppressed = false;
bool MessageDrop::fwkInfoAlwaysSuppressed = false;
bool MessageDrop::warningAlwaysSuppressed = false;
std::string MessageDrop::jobMode{};

MessageDrop* MessageDrop::instance() {
  thread_local static MessageDrop s_drop{};
  return &s_drop;
}
namespace {
  const std::string kBlankString{" "};
}

namespace edm {
  namespace messagedrop {

    class StringProducer {
    public:
      virtual ~StringProducer() {}
      virtual std::string theContext() const = 0;
    };

    class StringProducerWithPhase : public StringProducer {
    public:
      StringProducerWithPhase()
          : name_(&kBlankString),
            label_(&kBlankString),
            phasePtr_("(Startup)"),
            moduleID_(std::numeric_limits<unsigned int>::max()),
            cache_(),
            idLabelMap_() {}

      std::string theContext() const override {
        if (cache_.empty()) {
          if (moduleID_ != std::numeric_limits<unsigned int>::max()) {
            auto nameLableIter = idLabelMap_.find(moduleID_);
            if (nameLableIter != idLabelMap_.end()) {
              cache_.assign(nameLableIter->second);
              cache_.append(phasePtr_);
              return cache_;
            }
          }
          cache_.assign(*name_);
          cache_.append(":");
          cache_.append(*label_);
          idLabelMap_[moduleID_] = cache_;
          cache_.append(phasePtr_);
        }
        return cache_;
      }
      void set(std::string const& name, std::string const& label, unsigned int moduleID, const char* phase) {
        name_ = &name;
        label_ = &label;
        moduleID_ = moduleID;
        phasePtr_ = phase;
        cache_.clear();
      }

    private:
      std::string const* name_;
      std::string const* label_;
      const char* phasePtr_;
      unsigned int moduleID_;

      //This class is only used within a thread local object
      CMS_SA_ALLOW mutable std::string cache_;
      CMS_SA_ALLOW mutable std::map<unsigned int, std::string> idLabelMap_;
    };

    class StringProducerPath : public StringProducer {
    public:
      StringProducerPath()
          : typePtr_("PathNotYetEstablished")  // change log 4
            ,
            path_(" "),
            cache_() {}
      std::string theContext() const override {
        if (cache_.empty()) {
          cache_.assign(typePtr_);
          cache_.append(path_);
        }
        return cache_;
      }
      void set(const char* type, std::string const& pathname) {
        typePtr_ = type;
        path_ = pathname;
        cache_.clear();
      }

    private:
      const char* typePtr_;
      std::string path_;
      //This class is only used within a thread local object
      CMS_SA_ALLOW mutable std::string cache_;
    };

    class StringProducerSinglet : public StringProducer {
    public:
      StringProducerSinglet() : singlet_("(NoModuleName)") {}
      std::string theContext() const override { return singlet_; }
      void set(const char* sing) { singlet_ = sing; }

    private:
      const char* singlet_;
    };

  }  // namespace messagedrop

  MessageDrop::MessageDrop()
      : runEvent("pre-events"),
        streamID(std::numeric_limits<unsigned int>::max()),
        debugEnabled(true),
        infoEnabled(true),
        fwkInfoEnabled(true),
        warningEnabled(true),
        errorEnabled(true),
        spWithPhase(new messagedrop::StringProducerWithPhase),
        spPath(new messagedrop::StringProducerPath),
        spSinglet(new messagedrop::StringProducerSinglet),
        moduleNameProducer(spSinglet) {}

  MessageDrop::~MessageDrop() {
    delete spSinglet.get();
    delete spPath.get();
    delete spWithPhase.get();
  }

  void MessageDrop::setModuleWithPhase(std::string const& name,
                                       std::string const& label,
                                       unsigned int moduleID,
                                       const char* phase) {
    spWithPhase->set(name, label, moduleID, phase);
    moduleNameProducer = spWithPhase;
  }

  void MessageDrop::setPath(const char* type, std::string const& pathname) {
    spPath->set(type, pathname);
    moduleNameProducer = spPath;
  }

  void MessageDrop::setSinglet(const char* sing) {
    spSinglet->set(sing);
    moduleNameProducer = spSinglet;
  }

  std::string MessageDrop::moduleContext() { return moduleNameProducer->theContext(); }
  void MessageDrop::clear() { setSinglet(""); }
}  // namespace edm

unsigned char MessageDrop::messageLoggerScribeIsRunning = 0;
