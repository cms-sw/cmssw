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
// $Id: MessageDrop.cc,v 1.10 2010/11/02 21:04:01 fischler Exp $
//

// system include files
#include "boost/thread/tss.hpp"
#include <cstring>

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
    virtual void snapshot() = 0;
};
  
class StringProducerWithPhase : public StringProducer
{
  typedef std::map<const void*, std::string>::const_iterator NLMiter;
  public:
    StringProducerWithPhase() 
    : name_initial_value_  (" ")			// change log 4
    , label_initial_value_ (" ")			
    , name_                (&name_initial_value_)
    , label_    	   (&label_initial_value_)
    , phasePtr_ 	   ("(Startup)")
    , moduleID_            (0)
    {}
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
    virtual void snapshot() 				// change log 5
    {
      snapshot_name_ = *name_;
      name_ = &snapshot_name_;
      snapshot_label_ = *label_;
      label_ = &snapshot_label_;
      std::strncpy (snapshot_phase_,phasePtr_,PHASE_MAX_LENGTH);
      snapshot_phase_[PHASE_MAX_LENGTH] = 0;
      phasePtr_ = snapshot_phase_;
    }
  private:
    static const int PHASE_MAX_LENGTH = 32;
    std::string const name_initial_value_;
    std::string const label_initial_value_;
    std::string const * name_;
    std::string const * label_;
    const char* phasePtr_;
    const void * moduleID_;
    mutable std::string cache_;
    mutable std::map<const void*, std::string> nameLabelMap_;
    std::string snapshot_name_;
    std::string snapshot_label_;
    char snapshot_phase_[PHASE_MAX_LENGTH+1];
};

class StringProducerPath : public StringProducer{
  public:
    StringProducerPath() 
    : typePtr_("PathNotYetEstablished") 		// change log 4
    , path_ (" ") {}
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
    virtual void snapshot() 				// change log 5
    {
      std::strncpy (snapshot_type_,typePtr_,TYPE_MAX_LENGTH);
      snapshot_type_[TYPE_MAX_LENGTH] = 0;
      typePtr_ = snapshot_type_;
    }
  private:
    static const int TYPE_MAX_LENGTH = 32;
    const char* typePtr_;
    std::string path_;
    mutable std::string cache_;
    char snapshot_type_[TYPE_MAX_LENGTH+1];
};
  
class StringProducerSinglet : public StringProducer{
  public:
    StringProducerSinglet() : singlet_("(NoModuleName)") {}
    virtual std::string theContext() const {
      return singlet_;
    }
    void set(const char * sing) {singlet_ = sing; } 
    virtual void snapshot() 
    {
      std::strncpy (snapshot_singlet_,singlet_,SINGLET_MAX_LENGTH);
      snapshot_singlet_[SINGLET_MAX_LENGTH] = 0;
      singlet_ = snapshot_singlet_;
    }
  private:
    static const int SINGLET_MAX_LENGTH = 32;
    const char * singlet_;
    char snapshot_singlet_[SINGLET_MAX_LENGTH+1];
};

} // namespace messagedrop

MessageDrop::MessageDrop()
  : moduleName ("")
  , runEvent("pre-events")
  , jobreport_name()					
  , jobMode("")						
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

void MessageDrop::snapshot() {
  const_cast<messagedrop::StringProducer *>(moduleNameProducer)->snapshot();
}

} // namespace edm


unsigned char MessageDrop::messageLoggerScribeIsRunning = 0;
