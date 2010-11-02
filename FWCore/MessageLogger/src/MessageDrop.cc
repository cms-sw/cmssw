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
// $Id: MessageDrop.cc,v 1.9 2010/09/24 22:00:15 fischler Exp $
//

// system include files

#include "boost/thread/tss.hpp"

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

using namespace edm;


edm::Exception * MessageDrop::ex_p = 0;
bool MessageDrop::debugEnabled=true;
bool MessageDrop::infoEnabled=true;
bool MessageDrop::warningEnabled=true;
// The following are false at initialization (in case configure is not done)
// and are set true at the start of configure_ordinary_destinations, 
// but are set false once a destination is thresholded to react to the 
// corresponding severity: 
bool MessageDrop::debugAlwaysSuppressed=false;		// change log 2
bool MessageDrop::infoAlwaysSuppressed=false;	 	// change log 2
bool MessageDrop::warningAlwaysSuppressed=false; 	// change log 2

MessageDrop *
MessageDrop::instance()
{
  static boost::thread_specific_ptr<MessageDrop> drops;
  MessageDrop* drop = drops.get();
  if(drop==0) { 
    drops.reset(new MessageDrop);
    drop=drops.get(); 
  }
  return drop;
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
  typedef std::map<const void*, std::string>::const_iterator NLMiter;
  public:
    StringProducerWithPhase() : phasePtr_("PhaseNotYetFilled") {}
    virtual std::string theContext() const {
      if (cache_.empty()) {
	if (moduleID_ != 0) {
	  NLMiter nameLableIter = nameLabelMap_.find(moduleID_);
	  if  (nameLableIter != nameLabelMap_.end()) {
	    cache_.assign(nameLableIter->second);
	    cache_.append(phasePtr_);
	    return cache_; 
	  }
	}
	cache_.assign(*name_);
	cache_.append(":");
	cache_.append(*label_);
	nameLabelMap_[moduleID_] = cache_;
	cache_.append(phasePtr_);	
      }
      return cache_;
    }
    void set(std::string const & name,
  	     std::string const & label,
	     const void * moduleID,
	     const char* phase)  {
      name_ = &name;
      label_ = &label;     
      moduleID_ = moduleID;
      phasePtr_ = phase;
      cache_.clear();	     
    } 
  private:
    const char* phasePtr_;
    std::string const * name_;
    std::string const * label_;
    const void * moduleID_;
    mutable std::string cache_;
    mutable std::map<const void*, std::string> nameLabelMap_;
};

class StringProducerPath : public StringProducer{
  public:
    StringProducerPath() : typePtr_("PathNotYetFilled") {}
    virtual std::string theContext() const {
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
    StringProducerSinglet() : singlet_("") {}
    virtual std::string theContext() const {
      return singlet_;
    }
    void set(const char * sing) {singlet_ = sing; } 
  private:
    const char * singlet_;
};

} // namespace messagedrop


MessageDrop::MessageDrop()
  : moduleName ("")
  , runEvent("pre-events")
  , jobreport_name()					// change log 5
  , jobMode("")						// change log 6
  , spWithPhase(new  messagedrop::StringProducerWithPhase)
  , spPath (new messagedrop::StringProducerPath)
  , spSinglet (new messagedrop::StringProducerSinglet)
  , moduleNameProducer (spSinglet)
  {  } 

MessageDrop::~MessageDrop()
{
  delete spSinglet;
  delete spPath;
  delete spWithPhase;
}

void MessageDrop::setModuleWithPhase(std::string const & name,
  			  	     std::string const & label,
				     const void * moduleID,
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

} // namespace edm


unsigned char MessageDrop::messageLoggerScribeIsRunning = 0;
