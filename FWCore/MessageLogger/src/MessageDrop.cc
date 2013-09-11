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
#include "boost/thread/tss.hpp"
#include <cstring>
#include <limits>

// user include files
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// Change Log
//
// 1 12/13/07 mf     	the static drops had been file-global level; moved it
//		     	into the instance() method to cure a 24-byte memory
//			leak reported by valgrind. Suggested by MP.
//
// 2 9/23/10 mf		Variables supporting situations where no thresholds are
//                      low enough to react to LogDebug (or info, or warning)
//
// 3 11/2/10 mf, crj 	Prepare moduleContext method:
//			see MessageServer/src/MessageLogger.cc change 17.
//			Change is extensive, involving StringProducer and
//			its derivative classes.
//
// 4 11/29/10 mf	Intitalize all local string-holders in the various
//			string producers. 
//
// 5  mf 11/30/10	Snapshot method to prepare for invalidation of the   
//			pointers used to hold module context.  Supports 
//			surviving throws that cause objects to go out of scope.
//
// 6  mf 12/7/10	Fix in snapshot method to avoid strncpy from
//			a string to the identical address, which valgrind
// 			reports as an overlap problem.
//
// 7  fwyzard 7/6/11    Add support for discarding LogError-level messages
//                      on a per-module basis (needed at HLT)

using namespace edm;

// The following are false at initialization (in case configure is not done)
// and are set true at the start of configure_ordinary_destinations, 
// but are set false once a destination is thresholded to react to the 
// corresponding severity: 
bool MessageDrop::debugAlwaysSuppressed=false;		// change log 2
bool MessageDrop::infoAlwaysSuppressed=false;	 	// change log 2
bool MessageDrop::warningAlwaysSuppressed=false; 	// change log 2
std::string MessageDrop::jobMode{};

MessageDrop *
MessageDrop::instance()
{
  //needs gcc4.8.1
  //thread_local static s_drop{};
  //return &s_drop;
  
  static boost::thread_specific_ptr<MessageDrop> drops;
  MessageDrop* drop = drops.get();
  if(drop==0) { 
    drops.reset(new MessageDrop);
    drop=drops.get(); 
  }
  return drop;
  
}
namespace  {
  const std::string kBlankString{" "};
}

namespace edm {
namespace messagedrop {

class StringProducer {
  public:
    virtual ~StringProducer() {}
    virtual std::string theContext() const = 0;
};
  
class StringProducerWithPhase : public StringProducer
{
  public:
    StringProducerWithPhase() 
    : name_                (&kBlankString)
    , label_               (&kBlankString)
    , phasePtr_            ("(Startup)")
    , moduleID_            (std::numeric_limits<unsigned int>::max())
    , cache_               ()
    , idLabelMap_        ()
    {
    }

    virtual std::string theContext() const override {
      if (cache_.empty()) {
        if (moduleID_ != std::numeric_limits<unsigned int>::max()) {
          auto nameLableIter = idLabelMap_.find(moduleID_);
          if  (nameLableIter != idLabelMap_.end()) {
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
    void set(std::string const & name,
             std::string const & label,
             unsigned int moduleID,
             const char* phase)  {
      name_ = &name;
      label_ = &label;     
      moduleID_ = moduleID;
      phasePtr_ = phase;
      cache_.clear();	     
    } 
  private:
    std::string const * name_;
    std::string const * label_;
    const char* phasePtr_;
    unsigned int moduleID_;
    mutable std::string cache_;
    mutable std::map<unsigned int, std::string> idLabelMap_;
};

class StringProducerPath : public StringProducer{
  public:
    StringProducerPath() 
    : typePtr_("PathNotYetEstablished") 		// change log 4
    , path_ (" ")
    , cache_()
    {
    }
     virtual std::string theContext() const override {
      if ( cache_.empty() ) {
        cache_.assign(typePtr_);
        cache_.append(path_);
      }
      return cache_; 
    }
    void set(const char* type, std::string const & pathname) {
      typePtr_ = type;
      path_ = pathname;
      cache_.clear();	     
    } 
  private:
    const char* typePtr_;
    std::string path_;
    mutable std::string cache_;
};
  
class StringProducerSinglet : public StringProducer{
  public:
    StringProducerSinglet()
    : singlet_("(NoModuleName)")
    {
    }
    virtual std::string theContext() const override {
      return singlet_;
    }
    void set(const char * sing) {singlet_ = sing; } 
  private:
    const char * singlet_;
};

} // namespace messagedrop

MessageDrop::MessageDrop()
  : runEvent("pre-events")
  , spWithPhase(new  messagedrop::StringProducerWithPhase)
  , spPath (new messagedrop::StringProducerPath)
  , spSinglet (new messagedrop::StringProducerSinglet)
  , moduleNameProducer (spSinglet)
{ 
} 

MessageDrop::~MessageDrop()
{
  delete spSinglet;
  delete spPath;
  delete spWithPhase;
}

void MessageDrop::setModuleWithPhase(std::string const & name,
                                     std::string const & label,
                                     unsigned int moduleID,
                                     const char* phase) {
  spWithPhase->set(name, label, moduleID, phase);
  moduleNameProducer = spWithPhase;
}				     
 
void MessageDrop::setPath(const char* type, std::string const & pathname) {
  spPath->set(type, pathname);
  moduleNameProducer = spPath;
}

void MessageDrop::setSinglet(const char * sing) {
  spSinglet->set(sing);
  moduleNameProducer = spSinglet;
}

std::string MessageDrop::moduleContext() {
  return moduleNameProducer->theContext();
}
void MessageDrop::clear() {
  setSinglet("");
}
} // namespace edm


unsigned char MessageDrop::messageLoggerScribeIsRunning = 0;
