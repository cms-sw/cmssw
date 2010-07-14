// -*- C++ -*-
//
// Package:    LogErrorFilter
// Class:      LogErrorFilter
//
/**\class LogErrorFilter LogErrorFilter.cc FWCore/LogErrorFilter/src/LogErrorFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu Nov 12 15:59:28 CET 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class LogErrorFilter : public edm::EDFilter {
public:
  explicit LogErrorFilter(edm::ParameterSet const&);
  ~LogErrorFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual bool filter(edm::Event&, edm::EventSetup const&);

  // ----------member data ---------------------------
  edm::InputTag harvesterTag_;
  bool atLeastOneError_;
  bool atLeastOneWarning_;
  bool atLeastOneEntry_;
  std::vector<std::string> avoidCategories_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LogErrorFilter::LogErrorFilter(edm::ParameterSet const& iConfig) :
  harvesterTag_(iConfig.getParameter<edm::InputTag>("harvesterTag")),
  atLeastOneError_(iConfig.getParameter<bool>("atLeastOneError")),
  atLeastOneWarning_(iConfig.getParameter<bool>("atLeastOneWarning")),
  atLeastOneEntry_(atLeastOneError_ && atLeastOneWarning_),
  avoidCategories_(iConfig.getParameter<std::vector<std::string> >("avoidCategories")) {
  if (!atLeastOneError_ && !atLeastOneWarning_) {
    throw edm::Exception(edm::errors::Configuration) <<
      "Useless configuration of the error/warning filter. Need to select on an error or a warning or both.\n";
  }
}


LogErrorFilter::~LogErrorFilter() {

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
LogErrorFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<std::vector<edm::ErrorSummaryEntry> > errorsAndWarnings;
  iEvent.getByLabel(harvesterTag_,errorsAndWarnings);

  if (errorsAndWarnings.failedToGet()) {
    return false;
  } else {
    if (atLeastOneEntry_) {
      if (avoidCategories_.size() != 0) {
        for (uint iE = 0; iE != errorsAndWarnings->size(); ++iE) {
          //veto categories from user input.
          if (std::find(avoidCategories_.begin(),avoidCategories_.end(), ((*errorsAndWarnings)[iE]).category) != avoidCategories_.end()) {
	    continue;
	  } else {
	    return true;
	  }
	}
	return false;
      } else {
        return (errorsAndWarnings->size() != 0);
      }
    } else {
      if (atLeastOneError_ || atLeastOneWarning_) {
        uint nError = 0;
        uint nWarning = 0;
	for (uint iE = 0; iE != errorsAndWarnings->size(); ++iE) {
	  //veto categories from user input.
	  if (avoidCategories_.size() != 0) {
	    if (std::find(avoidCategories_.begin(),avoidCategories_.end(), ((*errorsAndWarnings)[iE]).category) != avoidCategories_.end()) {
	      continue;
	    }
	  }
	  edm::ELseverityLevel const& severity = ((*errorsAndWarnings)[iE]).severity;
	  //count errors
	  if (severity.getLevel() == edm::ELseverityLevel::ELsev_error || severity.getLevel() == edm::ELseverityLevel::ELsev_error2) {
	    ++nError;
	  }
	  //count warnings
	  if (severity.getLevel() == edm::ELseverityLevel::ELsev_warning || severity.getLevel() == edm::ELseverityLevel::ELsev_warning2) {
	    ++nWarning;
	  }
	}
	if (atLeastOneError_ && nError != 0) {
	  return (true);
	}
	if (atLeastOneWarning_ && nWarning != 0) {
	  return (true);
	}
      }
    }
  }
  return (false);
}


void
LogErrorFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("harvesterTag");
  desc.add<bool>("atLeastOneError");
  desc.add<bool>("atLeastOneWarning");
  desc.add<std::vector<std::string> >("avoidCategories");
  descriptions.add("logErrorFilter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LogErrorFilter);
