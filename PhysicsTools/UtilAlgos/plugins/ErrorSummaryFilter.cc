// -*- C++ -*-
//
// Package:    ErrorSummaryFilter
// Class:      ErrorSummaryFilter
//
/**\class ErrorSummaryFilter ErrorSummaryFilter.cc PhysicsTools/UtilAlgos/plugins/ErrorSummaryFilter.cc

 Description: Filter to remove events with given error types

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Benedikt Hegner
//         Created:  Thu May 10 20:23:28 CET 2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class ErrorSummaryFilter : public edm::global::EDFilter<> {
public:
  explicit ErrorSummaryFilter(edm::ParameterSet const&);

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<edm::ErrorSummaryEntry> > srcToken_;
  std::vector<std::string> modules_;
  std::string severityName_;
  std::vector<std::string> avoidCategories_;
  typedef std::vector<edm::ErrorSummaryEntry> ErrorSummaryEntries;
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
ErrorSummaryFilter::ErrorSummaryFilter(edm::ParameterSet const& iConfig)
    : srcToken_(consumes<std::vector<edm::ErrorSummaryEntry> >(iConfig.getParameter<edm::InputTag>("src"))),
      modules_(iConfig.getParameter<std::vector<std::string> >("modules")),
      severityName_(iConfig.getParameter<std::string>("severity")),
      avoidCategories_(iConfig.getParameter<std::vector<std::string> >("avoidCategories")) {
  if (!(severityName_ == "error" || severityName_ == "warning")) {
    throw edm::Exception(edm::errors::Configuration) << "Severity parameter needs to be 'error' or 'warning'.\n";
  }
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool ErrorSummaryFilter::filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  edm::Handle<std::vector<edm::ErrorSummaryEntry> > errorSummaryEntry;
  iEvent.getByToken(srcToken_, errorSummaryEntry);

  for (ErrorSummaryEntries::const_iterator i = errorSummaryEntry->begin(), end = errorSummaryEntry->end(); i != end;
       ++i) {
    if (std::find(modules_.begin(), modules_.end(), i->module) != modules_.end()) {
      if (std::find(avoidCategories_.begin(), avoidCategories_.end(), i->category) != avoidCategories_.end()) {
        continue;
      } else {
        edm::ELseverityLevel const& severity = i->severity;
        if (severityName_ == "error") {
          if (severity.getLevel() == edm::ELseverityLevel::ELsev_error ||
              severity.getLevel() == edm::ELseverityLevel::ELsev_warning) {
            return (false);
          }
        } else if (severityName_ == "warning") {
          if (severity.getLevel() == edm::ELseverityLevel::ELsev_warning) {
            return (false);
          }
        } else {
          continue;
        }
      }
    }
  }
  return (true);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ErrorSummaryFilter);
